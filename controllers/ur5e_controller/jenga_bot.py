from utils import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------
# Global robot configuration
# ------------------------------------------------------------

# Home joint configuration (with gripper open)
home_pos = [0, -2.1, 2.9, -2.8, -1.57, 0.0, True]


class RobotPlacerWithVision:
    """
    High‑level controller for a Webots robot that:
    - Locates Jenga blocks in the scene tree
    - Moves toward them using joint‑space interpolation
    - Performs nudging motions to adjust block positions
    """

    # Motion parameters
    max_joint_vel = 1.2     # rad/s
    DT = 0.016              # timestep (62.5 Hz)

    # State machine
    state = "move_search"
    target = "red"

    # Interpolation tracking
    target_q = None
    is_moving = False
    timeout = None

    # Tower geometry constants
    TOWER_CENTER = np.array([-0.062, 2.71, 0.715])
    BLOCK_WIDTH = 0.05
    BLOCK_LENGTH = 0.15
    BLOCK_HEIGHT = 0.03
    LAYER_HEIGHT = 0.03

    def __init__(self, robot=None):
        """
        Initialize the robot controller.

        Args:
            robot: Webots Supervisor instance for scene access
        """
        self.robot = robot
        self.phase = "find_next_block"
        self.layer = 1
        self.block_num = 2


    # ------------------------------------------------------------
    # Scene tree block lookup
    # ------------------------------------------------------------

    def get_block_position(self, layer, block_num):
        """
        Retrieve the world position of a block from Webots.

        Args:
            layer: Layer index (1–9)
            block_num: Block index (1–3)

        Returns:
            np.array([x, y, z]) or None if not found
        """
        if self.robot is None:
            return None

        block_name = f"solid({layer}{block_num})"

        try:
            root = self.robot.getRoot()
            children_field = root.getField("children")

            # Search entire scene tree recursively
            for i in range(children_field.getCount()):
                node = children_field.getMFNode(i)
                if node is None:
                    continue

                block_node = self._find_node_by_name(node, block_name)
                if block_node is not None:
                    translation_field = block_node.getField("translation")
                    if translation_field:
                        return np.array(translation_field.getSFVec3f())

            print(f"Warning: Block {block_name} not found")
            return None

        except Exception as e:
            print(f"Error retrieving block {block_name}: {e}")
            return None


    def _find_node_by_name(self, node, target_name):
        """
        Recursively search for a node by its internal name.

        Args:
            node: Node to search
            target_name: Name string to match

        Returns:
            Node or None
        """
        if node is None:
            return None

        # Check DEF name
        def_name = node.getField("name")
        if def_name is not None:
            if def_name.getSFString() == target_name:
                return node

        # Recursively search children
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


    # ------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------

    def limit_clamp(self, q):
        """Clamp joint angles to robot limits."""
        low = np.array([-2.9671, -2.0, -2.9671, -3.1416, -2.9671, -0.0873])
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
        q_next_full = getIK(0, desired_pose, current_q)

        # Smooth interpolation
        delta_q = q_next_full - current_q
        q_interpolated = current_q + rate_alpha * delta_q

        return limit_clamp(np.array(q_interpolated, dtype=float))


    def get_tower_top_height(self):
        """
        Compute the highest Z coordinate of any existing block.
        """
        max_height = self.TOWER_CENTER[2]

        for layer in range(1, 10):
            for block_num in range(1, 4):
                pos = self.get_block_position(layer, block_num)
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

        # Compute IK for the target pose (keeps configuration consistent)
        target_pose = self.q_to_pose(q_target)
        q_ik = getIK(0, target_pose, q_now)

        # Smooth interpolation
        q_next = q_now + rate * (q_ik - q_now)

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


    def step_to_target(self, cur_angles):
        """
        Perform one interpolation step toward target_q.
        """
        if self.target_q is None:
            return None

        max_step = self.max_joint_vel * self.DT
        new_angles = []
        all_reached = True

        for i in range(6):
            diff = self.target_q[i] - cur_angles[i]

            if abs(diff) < 0.001:
                new_angles.append(self.target_q[i])
            else:
                step = np.clip(diff, -max_step, max_step)
                new_angles.append(cur_angles[i] + step)
                all_reached = False

        if all_reached:
            self.is_moving = False

        gripper = self.target_q[6] if len(self.target_q) > 6 else False
        return new_angles + [gripper]


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
        cur_pose = forwardKinematics(cur_angles)
        cur_pos = cur_pose[:3, 3]

        # If currently interpolating, continue
        if self.is_moving and self.target_q is not None:
            return self.step_to_target(cur_angles)

        # Handle timeout
        if self.timeout is not None:
            if tt < self.timeout:
                return np.append(current_q, self.target_q[6])
            else:
                self.timeout = None

        return home_pos
