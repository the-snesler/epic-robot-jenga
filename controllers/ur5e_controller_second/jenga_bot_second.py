#FOR ROBO NUM 2

from utils import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


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
        self.phase = "align_first_guy"
        self.layer = 1
        self.block_num = 2

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
        
    def limit_clamp(self, q):
        low = np.array([-2.9671, -2.0, -2.9671, -3.1416, -2.9671, -0.0873])
        high = np.array([ 2.9671, 0.0, 2.9671, -0.4000, 2.9671, 3.8223])
        return np.clip(q, low, high)
        
    def q_to_pose(self, q):
        T = getFK(q)
        p = T[:3, 3]
        Rm = T[:3, :3]
        w = R.from_matrix(Rm).as_rotvec()
        return np.concatenate([p, w])


    def block_position_to_desired_pose(self, current_q, target_q, rate_alpha=0.12):
        desired_pose = self.q_to_pose(target_q)
        q_next_full = getIK(0, desired_pose, current_q)
        
        # Interpolate: move only rate_alpha fraction toward the full target joints
        delta_q = q_next_full - current_q
        q_interpolated = current_q + rate_alpha * delta_q
        
        return limit_clamp(np.array(q_interpolated, dtype=float))

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

    def move_to_joint_target(self, current_q_full, target_joint_angles, rate=0.15):
        """
        Smooth joint-space interpolation to a hardcoded (or any) joint target.
        Super stable – never flips configuration.
        
        current_q_full:      the 7-element list from Webots [q0..q5, gripper]
        target_joint_angles: list or array of 6 desired joint values
        rate:                how fast to move (0.05–0.25 is smooth and safe)
        """
        # Extract only the 6 joints
        q_now = np.array(current_q_full[:6], dtype=float)
        q_target = np.array(target_joint_angles, dtype=float)
    
        # Solve full IK at the target (to make sure it's reachable and pick the best solution)
        target_pose = self.q_to_pose(q_target)           # convert target joints → pose
        q_ik = getIK(0, target_pose, q_now)               # seed with current = stays in same config!
    
        # Smooth interpolation in joint space
        delta = q_ik - q_now
        q_next = q_now + rate * delta
    
        # Clamp to physical limits
        q_next = self.limit_clamp(q_next)
    
        return q_next
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
        # Uncomment this code if the robo gets stuck under da table again :(
        # print(tt)
        # if tt < 50:
            # return np.array([-2.94, -0.0, 0.5, 0, 0, 0, False])
        # elif tt < 100:
            # return np.array([-2.94, -2.00, 0.5, 0, 0, 0, False])
        if self.phase == "align_first_guy":
            print("align_first_guy")
            block_one_pose = np.array([-0.20, -2.00, 1.90, 3.37, -1.37, 0.00])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block"
            return np.append(result, 1).tolist()
            
        if self.phase == "go_to_next_block":
            print("go_to_next_block")
            block_one_pose = np.array([-0.25, -1.95, 2.10, 3.37, -1.37, 0.00])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_two"
            return np.append(result, 0).tolist()
            
        if self.phase == "go_to_next_block_two":
            print("go_to_next_block_two")
            block_one_pose = np.array([-0.25, -1.60, 1.90, 1.00, -1.37, 0.00])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_three"
            return np.append(result, 0).tolist()
            
        if self.phase == "go_to_next_block_three":
            print("go_to_next_block_three")
            block_one_pose = np.array([-0.25, -1.45, 1.80, 1.00, -1.37, 0.00])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_four"
            return np.append(result, 1).tolist()
            
        if self.phase == "go_to_next_block_four":
            print("go_to_next_block_four")
            block_one_pose = np.array([-0.25, -1.56, 1.84, 1.50, -1.34, 1.50])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_five"
            return np.append(result, 0).tolist()
            
        if self.phase == "go_to_next_block_five":
            print("go_to_next_block_five")
            block_one_pose = np.array([-0.25, -1.56, 1.84, 1.50, -1.34, 1.50])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all() and tt > 250:
                self.phase = "go_to_next_block_six"
            return np.append(result, 1).tolist()
            
        if self.phase == "go_to_next_block_six":
            print("go_to_next_block_six")
            block_one_pose = np.array([-0.25, -2.10, 2.24, 1.00, -1.39, 1.50])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_seven"
            return np.append(result, 1).tolist()
            
        if self.phase == "go_to_next_block_seven":
            print("go_to_next_block_seven")
            block_one_pose = np.array([-0.25, -2.70, 2.10, 1.00, -1.39, 1.50])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_eight"
            return np.append(result, 1).tolist()
        if self.phase == "go_to_next_block_eight":
            print("go_to_next_block_eight")
            block_one_pose = np.array([-0.20, -2.00, 1.90, 3.37, -1.37, 0.00])
            result = self.move_to_joint_target(current_q, block_one_pose)
            
            q_diff = abs(current_q - result)
            if (q_diff < 0.0001).all():
                self.phase = "go_to_next_block_eight"
            return np.append(result, 1).tolist()

        # done or error
        gripper_state = (
            self.target_q[6]
            if self.target_q is not None and len(self.target_q) >= 7
            else False
        )
        return np.append(current_q[:6], gripper_state)
