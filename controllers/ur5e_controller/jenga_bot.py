from utils import *
import numpy as np
import math

home_pos = [0, 0, 0, -math.pi/2, 0, 0, False]  # home position with gripper open

class RobotPlacerWithVision:
    max_joint_vel = 1.2 # radians per second
    DT = 0.016  # seconds (62.5 Hz)

    state = "move_search" # other: search, pick, move_place, place, done
    target = "red" # other: blue

    # Interpolation tracking
    target_q = None
    is_moving = False
    timeout = None

    # Tower geometry constants
    TOWER_CENTER = np.array([-0.062, 2.71, 0.715])  # World coordinates
    BLOCK_WIDTH = 0.05  # 50mm
    BLOCK_LENGTH = 0.15  # 150mm
    BLOCK_HEIGHT = 0.03  # 30mm
    LAYER_HEIGHT = 0.03  # 30mm per layer

    def __init__(self, robot=None):
        """Initialize the robot controller

        Args:
            robot: Webots Supervisor instance for accessing scene objects
        """
        self.robot = robot

    def get_block_position(self, layer, block_num):
        """Get the current position of a block from Webots

        Args:
            layer: Layer number (1-9)
            block_num: Block number in layer (1-3)

        Returns:
            np.array([x, y, z]) position or None if block not found
        """
        if self.robot is None:
            return None

        block_name = f"solid({layer}{block_num})"

        # Try to get the block node from the scene tree
        # Since blocks don't have DEF names, we need to search by name
        try:
            # Get root node and search children
            root = self.robot.getRoot()
            children_field = root.getField("children")

            for i in range(children_field.getCount()):
                node = children_field.getMFNode(i)
                if node is None:
                    continue

                # Search recursively in the scene tree
                block_node = self._find_node_by_name(node, block_name)
                if block_node is not None:
                    translation_field = block_node.getField("translation")
                    if translation_field:
                        pos = translation_field.getSFVec3f()
                        return np.array(pos)

            print(f"Warning: Block {block_name} not found in scene")
            return None

        except Exception as e:
            print(f"Error getting block position for {block_name}: {e}")
            return None

    def _find_node_by_name(self, node, target_name):
        """Recursively search for a node by checking its DEF or internal name

        Args:
            node: Current node to search
            target_name: Name to search for (e.g., "solid(11)")

        Returns:
            Node if found, None otherwise
        """
        if node is None:
            return None

        # Check DEF name
        def_name = node.getField("name")
        if def_name is not None:
            def_name = def_name.getSFString()
            if def_name == target_name:
                return node

        # Check if node has children field
        try:
            children_field = node.getField("children")
            if children_field:
                for i in range(children_field.getCount()):
                    child = children_field.getMFNode(i)
                    result = self._find_node_by_name(child, target_name)
                    if result is not None:
                        return result
        except:
            pass

        return None

    def get_tower_top_height(self):
        """Calculate the Z-coordinate of the top of the tower

        Returns:
            float: Z-coordinate of tower top, or estimated value if blocks not accessible
        """
        # Try to find the highest block that still exists
        max_height = self.TOWER_CENTER[2]  # Base height

        for layer in range(1, 10):  # Layers 1-9
            for block_num in range(1, 4):  # Blocks 1-3
                pos = self.get_block_position(layer, block_num)
                if pos is not None:
                    block_top = pos[2] + self.BLOCK_HEIGHT / 2
                    max_height = max(max_height, block_top)

        return max_height

    def set_target(self, target_angles, current_angles):
        """Set a new movement target and start interpolation"""
        self.target_q = target_angles.copy()
        self.is_moving = True

    def set_speed(self, speed):
        """Set the maximum joint velocity (radians per second)"""
        self.max_joint_vel = speed

    def set_timeout(self, timeout_tt):
        """Set a timeout timestep to block new commands until after that time"""
        self.timeout = timeout_tt

    def step_to_target(self, cur_angles):
        if self.target_q is not None:
            # Calculate max step per joint based on velocity limit
            max_step = self.max_joint_vel * self.DT

            # Interpolate each joint angle
            new_angles = []
            all_reached = True
            for i in range(6):
                current_diff = self.target_q[i] - cur_angles[i]

                # If we're close enough to target, snap to it
                if abs(current_diff) < 0.001:
                    new_angles.append(self.target_q[i])
                else:
                    # Move towards target, but don't exceed max velocity
                    step = np.clip(current_diff, -max_step, max_step)
                    new_angles.append(cur_angles[i] + step)
                    all_reached = False

            if all_reached:
                self.is_moving = False

            gripper_state = self.target_q[6] if len(self.target_q) > 6 else False
            return new_angles + [gripper_state]

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        cur_angles = current_q.copy()[0:6]
        cur_pose = forwardKinematics(cur_angles)
        cur_pos = cur_pose[:3,3]
        # print("Current state:", self.state, "at position", cur_pos, "is_moving:", self.is_moving)
        # print("Current angles:", cur_angles)

        # Handle interpolation if we're currently moving
        if self.is_moving and self.target_q is not None:
            return self.step_to_target(cur_angles)

        if self.timeout is not None and tt < self.timeout:
            return np.append(current_q, self.target_q[6]) # type: ignore

        if self.timeout is not None and tt >= self.timeout:
            self.timeout = None

        # State machine logic

        # done or error
        gripper_state = (
            self.target_q[6]
            if self.target_q is not None and len(self.target_q) >= 7
            else False
        )
        return np.append(current_q[:6], gripper_state)
