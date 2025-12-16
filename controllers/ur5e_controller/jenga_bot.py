from utils import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

home_pose = [0, -2.1, 2.1, -3.0, -1.57, 0.0, True]
robot_world_coords = np.array([0.0, 2.2, 0.3, 0.0, 0.0, 1.0, 1.57085]) # x, y, z, rotvec

class RobotPlacerWithVision:
    """
    High‑level controller for a Webots robot that:
    - Locates Jenga blocks in the scene tree
    - Moves toward them using joint‑space interpolation
    - Performs nudging motions to adjust block positions
    """

    # Motion parameters
    max_joint_vel = 0.5     # rad/s
    DT = 0.016              # timestep (62.5 Hz)

    # State machine
    state = "move_search"
    target = "red"

    # Interpolation tracking
    target_q = None
    is_moving = False
    timeout = None

    # Tower geometry constants
    BLOCK_WIDTH = 0.05
    BLOCK_LENGTH = 0.17
    BLOCK_HEIGHT = 0.05
    LAYER_HEIGHT = 0.05

    def __init__(self, robot=None):
        """
        Initialize the robot controller.

        Args:
            robot: Webots Supervisor instance for scene access
        """
        self.robot = robot
        self.phase = "find_next_block"
        self.layer = 3
        self.block_num = 1


    # ------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------

    def limit_clamp(self, q):
        """Clamp joint angles to robot limits."""
        low = np.array([-2.9671, -2.0, -2.9671, -4.1416, -2.9671, -0.0873])
        high = np.array([2.9671, 0.0, 2.9671, -0.4000, 2.9671, 3.8223])
        return np.clip(q, low, high)

    def q_to_pose(self, q):
        """Convert joint angles to pose [x,y,z,rotvec]."""
        T = getFK(q)
        p = T[:3, 3]
        Rm = T[:3, :3]
        w = R.from_matrix(Rm).as_rotvec()
        return np.concatenate([p, w])


    def block_position_to_desired_pose(self, current_q, target_q, rate_alpha=0.12):
        """
        Compute a smoothed IK step toward a target pose.
        """
        desired_pose = self.q_to_pose(target_q)
        q_next_full = getIK(desired_pose, current_q)

        # Smooth interpolation
        delta_q = q_next_full - current_q
        q_interpolated = current_q + rate_alpha * delta_q

        return self.limit_clamp(np.array(q_interpolated, dtype=float))


    def get_tower_top_height(self):
        """
        Compute the highest Z coordinate of any existing block.
        """
        max_height = 0
        if self.robot is None:
            return max_height

        for layer in range(1, 10):
            for block_num in range(1, 4):
                pos = get_block_position(self.robot.getRoot(), layer, block_num)
                if pos is not None:
                    block_top = pos[2] + self.BLOCK_HEIGHT / 2
                    max_height = max(max_height, block_top)

        return max_height


    # ------------------------------------------------------------
    # Joint‑space interpolation
    # ------------------------------------------------------------

    def move_to_joint_target(self, current_q_full, target_joint_angles, rate=0.15):
        """
        Smoothly interpolate toward a target joint configuration.

        Args:
            current_q_full: 7‑element list [q0..q5, gripper]
            target_joint_angles: 6‑element target joint vector
            rate: interpolation factor (0.05–0.25)
        """
        q_now = np.array(current_q_full[:6], dtype=float)
        q_target = np.array(target_joint_angles, dtype=float)

        # Smooth interpolation
        q_next = q_now + rate * (q_target - q_now)

        return self.limit_clamp(q_next)


    def set_target(self, target_angles, current_angles):
        """Begin interpolation toward a new target."""
        self.target_q = target_angles.copy()
        self.is_moving = True

    def set_speed(self, speed):
        """Set max joint velocity."""
        self.max_joint_vel = speed

    def set_timeout(self, timeout_tt):
        """Block new commands until a future timestep."""
        self.timeout = timeout_tt


    # ------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------

    def getRobotCommand(self, tt, current_q, current_image_bgr):
        """
        Main state machine for robot control.

        Args:
            tt: timestep
            current_q: current joint configuration
            current_image_bgr: camera image (unused)
        """
        cur_angles = current_q.copy()[0:6]
        cur_pose = getFK(cur_angles)
        cur_pos = cur_pose[:3, 3]

        # Handle timeout
        if self.timeout is not None:
            if tt < self.timeout:
                return np.append(current_q, True).tolist()
            else:
                self.timeout = None
        
        if self.robot is None:
            return np.append(home_pose, True).tolist()

        print("controller 1 tt: " + str(tt) + " phase: " + str(self.phase))
    
        if self.phase == "find_next_block":
            # Get position of next block in layer 3
            block_pos = get_block_position(self.robot.getRoot(), 3, self.block_num)
            
            if block_pos is None:
                print(f"Block 3-{self.block_num} not found, skipping")
                self.block_num += 1
                if self.block_num > 3:
                    self.phase = "done"
                return np.append(current_q[:6], False)
            
            # Transform block position from world to robot frame
            # Extract robot world position and rotation
            robot_pos = robot_world_coords[:3]
            robot_rotvec = np.array(robot_world_coords[3:6]) * robot_world_coords[6]
            
            # Create rotation matrix from rotation vector
            rot = R.from_rotvec(robot_rotvec)
            
            # Transform: rotate inverse and translate
            block_in_robot_frame = rot.inv().apply(block_pos - robot_pos)
            self.target_block_pos = block_in_robot_frame
            
            print(f"Found block 3-{self.block_num} at world {block_pos}, robot frame {self.target_block_pos}")
            self.phase = "approach_block"


        if self.phase == "approach_block":
            # Position around
            approach_pos = self.target_block_pos.copy()
            approach_pos[0] -= 0.28
            approach_pos[2] -= 0.093
            
            # Create pose with downward orientation
            approach_pose = np.concatenate([approach_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(approach_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.12)
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "nudge_block"
                # self.block_num += 1
                # if self.block_num > 3:
                #     self.phase = "done"
                # else:
                #     self.phase = "find_next_block"
        
            return np.append(result, True).tolist()

        elif self.phase == "nudge_block":
            # Push through block position
            nudge_pos = self.target_block_pos.copy()
            nudge_pos[0] -= 0.28
            nudge_pos[0] += 0.065
            nudge_pos[2] -= 0.093
            
            nudge_pose = np.concatenate([nudge_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(nudge_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.05)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "retract"
            
            return np.append(result, True).tolist()

       
        elif self.phase == "retract":
            # Move back to approach position
            retract_pos = self.target_block_pos.copy()
            retract_pos[0] -= 0.28
            retract_pos[2] -= 0.093
            
            retract_pose = np.concatenate([retract_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(retract_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.12)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.block_num += 2
                if self.block_num > 3:
                    self.phase = "done"
                else:
                    self.phase = "find_next_block"
            
            return np.append(result, True).tolist()

        elif self.phase == "done":
            return np.append(current_q[:6], False)
