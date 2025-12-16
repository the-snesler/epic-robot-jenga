# Import necessary modules
from utils import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# Home position with gripper open
home_pos = [0, -2.1, 2.1, -3.0, -1.57, 0.0, True]

class RobotPlacerWithVision:
    # Class constants
    max_joint_vel = 1.2  # Maximum joint velocity in radians per second
    DT = 0.016  # Time step (62.5 Hz)
    
    # Initial state setup
    state = "move_search"  # Possible states: search, pick, move_place, place, done
    target = "red"  # Target color: "red" or "blue"

    # Interpolation tracking variables
    target_q = None
    is_moving = False
    timeout = None

    # Tower geometry constants
    TOWER_CENTER = np.array([-0.062, 2.71, 0.715])  # World coordinates for the tower center
    BLOCK_WIDTH = 0.05  # Width of a block (50mm)
    BLOCK_LENGTH = 0.15  # Length of a block (150mm)
    BLOCK_HEIGHT = 0.03  # Height of a block (30mm)
    LAYER_HEIGHT = 0.03  # Height per layer of blocks (30mm)

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
            np.array([x, y, z]): Position of the block or None if block is not found
        """
        if self.robot is None:
            return None

        block_name = f"solid({layer}{block_num})"

        # Try to get the block node from the scene tree
        try:
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

        # Check if node has children field and search recursively
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
        """Clamp joint values to the robot's physical limits
        
        Args:
            q: Array of joint angles to clamp
        
        Returns:
            np.array: Clamped joint angles
        """
        low = np.array([-2.9671, -2.0, -2.9671, -3.1416, -2.9671, -0.0873])
        high = np.array([2.9671, 0.0, 2.9671, -0.4000, 2.9671, 3.8223])
        return np.clip(q, low, high)

    def q_to_pose(self, q):
        """Convert joint angles to pose (position + rotation vector)
        
        Args:
            q: Array of joint angles
        
        Returns:
            np.array: Position and rotation vector (3D position + 3D rotation)
        """
        T = getFK(q)  # Forward kinematics for the joint angles
        p = T[:3, 3]  # Extract position from transformation matrix
        Rm = T[:3, :3]  # Extract rotation matrix
        w = R.from_matrix(Rm).as_rotvec()  # Convert rotation matrix to rotation vector
        return np.concatenate([p, w])  # Combine position and rotation vector

    def block_position_to_desired_pose(self, current_q, target_q, rate_alpha=0.12):
        """Interpolate between the current and target joint configurations
        
        Args:
            current_q: Current joint configuration
            target_q: Target joint configuration
            rate_alpha: Interpolation rate (between 0 and 1)
        
        Returns:
            np.array: Interpolated joint angles
        """
        desired_pose = self.q_to_pose(target_q)  # Convert target joint angles to pose
        q_next_full = getIK(0, desired_pose, current_q)  # Solve IK to get next joint angles
        
        # Interpolate towards the full target joint configuration
        delta_q = q_next_full - current_q
        q_interpolated = current_q + rate_alpha * delta_q
        
        return self.limit_clamp(np.array(q_interpolated, dtype=float))  # Return clamped interpolated angles

    def get_tower_top_height(self):
        """Calculate the Z-coordinate of the top of the tower
        
        Returns:
            float: Z-coordinate of the tower top, or estimated value if blocks not accessible
        """
        max_height = self.TOWER_CENTER[2]  # Base height of the tower

        # Loop through each layer and block to find the highest block
        for layer in range(1, 10):  # Layers 1-9
            for block_num in range(1, 4):  # Blocks 1-3
                pos = self.get_block_position(layer, block_num)
                if pos is not None:
                    block_top = pos[2] + self.BLOCK_HEIGHT / 2
                    max_height = max(max_height, block_top)

        return max_height

    def move_to_joint_target(self, current_q_full, target_joint_angles, rate=0.15):
        """Smoothly move the robot's joints to the target configuration
        
        Args:
            current_q_full: Current joint angles (including gripper state)
            target_joint_angles: Target joint angles (without gripper state)
            rate: Rate of movement (0.05–0.25 is smooth and safe)
        
        Returns:
            np.array: Next joint configuration
        """
        q_now = np.array(current_q_full[:6], dtype=float)  # Get only the 6 joint angles
        q_target = np.array(target_joint_angles, dtype=float)

        # Solve full IK at the target to ensure it's reachable and pick the best solution
        target_pose = self.q_to_pose(q_target)  # Convert target joint angles to pose
        q_ik = getIK(0, target_pose, q_now)  # Inverse kinematics

        # Smooth interpolation in joint space
        delta = q_ik - q_now
        q_next = q_now + rate * delta

        # Clamp to physical limits
        q_next = self.limit_clamp(q_next)

        return q_next

    def set_target(self, target_angles, current_angles):
        """Set a new target joint configuration for movement
        
        Args:
            target_angles: Target joint angles
            current_angles: Current joint angles (for interpolation)
        """
        self.target_q = target_angles.copy()
        self.is_moving = True

    def set_speed(self, speed):
        """Set the maximum joint velocity (radians per second)
        
        Args:
            speed: Maximum joint velocity in radians per second
        """
        self.max_joint_vel = speed

    def set_timeout(self, timeout_tt):
        """Set a timeout to block new commands until the specified time
        
        Args:
            timeout_tt: Timeout time step
        """
        self.timeout = timeout_tt

    def step_to_target(self, cur_angles):
        """Move the robot step by step toward the target joint angles
        
        Args:
            cur_angles: Current joint angles
        
        Returns:
            np.array: Next joint angles (including gripper state)
        """
        if self.target_q is not None:
            max_step = self.max_joint_vel * self.DT  # Max step per joint based on velocity limit

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
                self.is_moving = False  # Stop moving if all joints are at the target

            # Handle the gripper state (if included in target_q)
            gripper_state = self.target_q[6] if len(self.target_q) > 6 else False
            return new_angles + [gripper_state]

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        """Get the robot's command based on the current state and time step
        
        Args:
            tt: Current time step
            current_q: Current joint angles (including gripper state)
            current_image_bgr: Current image captured by the robot's camera
        
        Returns:
            np.array: New joint angles and gripper state
        """
        cur_angles = current_q.copy()[:6]  # Get the 6 joint angles (excluding gripper)
        cur_pose = forwardKinematics(cur_angles)  # Compute forward kinematics
        cur_pos = cur_pose[:3, 3]  # Extract the position from the forward kinematics

        # Handle interpolation if the robot is currently moving
        if self.is_moving and self.target_q is not None:
            return self.step_to_target(cur_angles)

        # If a timeout is set, prevent new commands until the timeout has passed
        if self.timeout is not None and tt < self.timeout:
            return np.append(current_q, self.target_q[6])  # Return the current joint angles and gripper state

        # Reset the timeout if the time step has exceeded the timeout
        if self.timeout is not None and tt >= self.timeout:
            self.timeout = None

        # go to home pos
        return home_pos
