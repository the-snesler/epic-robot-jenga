from controllers.ur5e_controller.utils import *
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

    def __init__(self):
        pass

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
        gripper_state = self.target_q[6] if self.target_q is not None and len(self.target_q) > 6 else current_q[6]
        return np.append(current_q[:6], gripper_state)
