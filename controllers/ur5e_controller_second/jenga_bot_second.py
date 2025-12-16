from utils import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

home_pose = [0, -2.1, 2.1, -3.0, -1.57, 0.0, True]
robot_world_coords = np.array([-0.100074, 3.67, 0.27, 0.0, 0.0, 1.0, -1.5707453]) # x, y, z, rotvec

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
    grab_timer = None
    release_timer = 0
    nudge_timer = 0

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
        
        # Simulated grab state
        self.grabbed_block_node = None
        self.block_offset_robot_frame = None
        self.grabbed_block_initial_pos = None
        self.block_physics_node = None  # Store original physics node
        
        # Phase timers
        self.release_timer = 0
        self.nudge_timer = 0


    # ------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------


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

        return np.array(q_interpolated, dtype=float)


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


    def disable_block_physics(self, block_node):
        """
        Disable physics simulation for a block to prevent collisions.
        Removes the Physics node to make the block intangible.
        
        Args:
            block_node: Webots node reference
        """
        if block_node is None:
            return
        
        try:
            # Get the physics field
            physics_field = block_node.getField("physics")
            if physics_field is None:
                print("Block has no physics field")
                return
            
            # Store the original physics node for restoration later
            self.block_physics_node = physics_field.getSFNode()
            
            if self.block_physics_node is not None:
                # Remove the physics node (makes block intangible)
                physics_field.removeSF()
                print("Disabled physics for grabbed block")
            else:
                print("Block already has no physics")
        except Exception as e:
            print(f"Could not disable physics: {e}")
    
    def enable_block_physics(self, block_node):
        """
        Re-enable physics simulation for a block.
        Restores the Physics node that was removed earlier.
        
        Args:
            block_node: Webots node reference
        """
        if block_node is None:
            return
        
        try:
            physics_field = block_node.getField("physics")
            if physics_field is None:
                print("Block has no physics field")
                return
            
            # Restore the physics node if we saved it
            if self.block_physics_node is not None:
                # Export the physics node as string and re-import it
                physics_string = self.block_physics_node.exportString()
                physics_field.importSFNodeFromString(physics_string)
                self.block_physics_node = None
                print("Re-enabled physics for released block")
            else:
                print("No physics node to restore")
        except Exception as e:
            print(f"Could not enable physics: {e}")
    
    def update_grabbed_block_position(self, current_q):
        """
        Update the position of the grabbed block to maintain offset from gripper.
        
        Args:
            current_q: Current joint configuration
        """
        if self.grabbed_block_node is None or self.block_offset_robot_frame is None:
            return
        
        # Get current gripper position in robot frame
        T = getFK(current_q[:6])
        gripper_pos_robot = T[:3, 3]
        
        # Add offset to get block position in robot frame
        block_pos_robot = gripper_pos_robot + self.block_offset_robot_frame
        
        # Transform to world coordinates
        robot_pos = robot_world_coords[:3]
        robot_rotvec = np.array(robot_world_coords[3:6]) * robot_world_coords[6]
        rot = R.from_rotvec(robot_rotvec)
        
        block_pos_world = robot_pos + rot.apply(block_pos_robot)
        
        # Update block position in Webots
        translation_field = self.grabbed_block_node.getField("translation")
        if translation_field:
            translation_field.setSFVec3f(block_pos_world.tolist())


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

        return q_next


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
        
        if self.robot is None:
            return np.append(home_pose, True).tolist()

        print("controller 2 tt: " + str(tt) + " phase: " + str(self.phase))
    
        if self.phase == "find_next_block":
            # Get position of next block in layer 3
            block_pos = get_block_position(self.robot.getRoot(), 3, self.block_num)
            
            if block_pos is None:
                print(f"Block 3-{self.block_num} not found, skipping")
                self.block_num += 1
                if self.block_num > 3:
                    self.phase = "done"
                return np.append(current_q[:6], False)
            
            # Get block node reference for later manipulation
            self.grabbed_block_node = get_block_node(self.robot.getRoot(), 3, self.block_num)
            self.grabbed_block_initial_pos = block_pos.copy()
            
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
            approach_pos[0] -= 0.27
            approach_pos[2] -= 0.093
            
            # Create pose with downward orientation
            approach_pose = np.concatenate([approach_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(approach_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.12)
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "grab_block"
        
            return np.append(result, False).tolist()

        elif self.phase == "grab_block":
            if not self.grab_timer or tt >= self.grab_timer + 100:
                self.grab_timer = tt + 50
                return np.append(current_q, False).tolist()
            elif tt > self.grab_timer + 40:
                # Calculate offset from gripper to block in robot frame
                T = getFK(current_q[:6])
                gripper_pos_robot = T[:3, 3]
                self.block_offset_robot_frame = self.target_block_pos - gripper_pos_robot
                print(f"Calculated block offset: {self.block_offset_robot_frame}")
                
                # Disable physics to prevent collisions during manipulation
                self.disable_block_physics(self.grabbed_block_node)
                
                self.phase = "retract"
                return np.append(current_q, True).tolist()
            elif tt > self.grab_timer:
                return np.append(current_q, True).tolist()
            else:
                return np.append(current_q, False).tolist()
       
        elif self.phase == "retract":
            # Update grabbed block position to follow gripper
            self.update_grabbed_block_position(current_q)
            
            # Move back to approach position
            retract_pos = self.target_block_pos.copy()
            retract_pos[0] -= 0.4
            retract_pos[2] -= 0.093
            
            retract_pose = np.concatenate([retract_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(retract_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.08)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "place_on_tower"
            
            return np.append(result, True).tolist()

        elif self.phase == "place_on_tower":
            # Phase 1: Move up vertically to approach position above tower
            # Update grabbed block position
            self.update_grabbed_block_position(current_q)
            
            # Move to position above original location (vertical approach)
            place_pos = self.target_block_pos.copy()
            place_pos[0] -= 0.4
            place_pos[2] += 0
            
            place_pose = np.concatenate([place_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(place_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.08)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "place_insert"
            
            return np.append(result, True).tolist()

        elif self.phase == "place_insert":
            # Phase 2: Move horizontally to insert block onto tower
            # Update grabbed block position
            self.update_grabbed_block_position(current_q)
            
            # Move horizontally to placement position (same x/y as original, higher z)
            insert_pos = self.target_block_pos.copy()
            insert_pos[0] -= 0.28
            insert_pos[2] += 0
            
            insert_pose = np.concatenate([insert_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(insert_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.06)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "place_release"
                self.release_timer = tt
            
            return np.append(result, True).tolist()

        elif self.phase == "place_release":
            # Phase 3: Release the block
            # Update grabbed block position while gripper is still closed
            if tt < self.release_timer + 20:
                self.update_grabbed_block_position(current_q)
                return np.append(current_q, False).tolist()
            
            # Re-enable physics before releasing
            self.enable_block_physics(self.grabbed_block_node)
            
            # Release the block
            self.grabbed_block_node = None
            self.block_offset_robot_frame = None
            print(f"Released block 3-{self.block_num}")
            
            self.phase = "place_backup"
            return np.append(current_q, False).tolist()

        elif self.phase == "place_backup":
            # Phase 4: Back up and close gripper
            # Move back slightly
            backup_pos = self.target_block_pos.copy()
            backup_pos[0] -= 0.4
            backup_pos[2] += 0
            
            backup_pose = np.concatenate([backup_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(backup_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.08)
            
            if np.all(np.abs(current_q[:6] - result) < 0.001):
                self.phase = "place_nudge"
                self.nudge_timer = tt
            
            return np.append(result, True).tolist()

        elif self.phase == "place_nudge":
            # Phase 5: Push block into final position
            # Move forward to nudge the block
            nudge_pos = self.target_block_pos.copy()
            nudge_pos[0] -= 0.25
            nudge_pos[2] += 0
            
            nudge_pose = np.concatenate([nudge_pos, R.from_euler('xyz', [0, -80, 0], degrees=True).as_rotvec()])
            q_target = getIK(nudge_pose, cur_angles)
            q_target[5] = 0.0
            
            result = self.move_to_joint_target(current_q, q_target, rate=0.05)
            
            # Hold nudge position briefly
            if np.all(np.abs(current_q[:6] - result) < 0.001) and tt > self.nudge_timer + 30:
                # Move to next block
                self.block_num += 2
                if self.block_num > 3:
                    self.phase = "done"
                else:
                    self.phase = "find_next_block"
            
            return np.append(result, True).tolist()

        elif self.phase == "done":
            return np.append(current_q[:6], False)
