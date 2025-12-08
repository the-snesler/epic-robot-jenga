from utils import *
import numpy as np
import math

home_pos = [0, -1.4, 1.2, -2.0, -1.57, 1.03]  # home joint angles (not position)

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
    TOWER_CENTER = np.array([2.54, 0.062, 0.715])  # World coordinates
    # to make them work, we had to switch X and Y, then reflect Y
    ROBOT_BASE = np.array([2.2, 0, 0.7])
    BLOCK_WIDTH = 0.05  # 50mm
    BLOCK_LENGTH = 0.15  # 150mm
    BLOCK_HEIGHT = 0.03  # 30mm
    LAYER_HEIGHT = 0.03  # 30mm per layer
    BLOCK_SPACING = 0.052  # 52mm center-to-center

    # Motion parameters
    PUSH_DEPTH = 0.035  # 35mm push distance
    GRIP_WIDTH = 0.045  # 45mm grip width for 50mm blocks
    APPROACH_OFFSET = 0.08  # 80mm offset for approach
    RETRACT_OFFSET = 0.08  # 80mm offset for retract
    SAFE_HEIGHT = 0.15  # 150mm above tower for transit
    PLACE_HEIGHT_OFFSET = 0.05  # 50mm above layer for placement approach

    # Safety parameters
    TOWER_CLEARANCE_RADIUS = 0.25  # 25cm clearance around tower center
    COLLISION_MARGIN = 0.02  # 20mm safety margin

    def __init__(self, robot=None):
        """Initialize the robot controller

        Args:
            robot: Webots Supervisor instance for accessing scene objects
        """
        self.robot = robot

        # Track removed blocks and touched layers
        self.removed_blocks = set()  # Set of (layer, block_num) tuples
        self.touched_layers = set()  # Set of layer numbers that have had blocks removed
        self.blocks_placed_on_top = 0  # Count of blocks placed on top layer

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
        self.target_q = np.array(target_angles, dtype=float)
        cur_q = np.array(current_angles, dtype=float)

        # Ensure we take the shortest path to the target angles
        for i in range(min(6, len(self.target_q))):
            diff = self.target_q[i] - cur_q[i]
            shortest_diff = (diff + np.pi) % (2 * np.pi) - np.pi
            self.target_q[i] = cur_q[i] + shortest_diff

        self.is_moving = True

    def set_speed(self, speed):
        """Set the maximum joint velocity (radians per second)"""
        self.max_joint_vel = speed

    def set_timeout(self, timeout_tt):
        """Set a timeout timestep to block new commands until after that time"""
        self.timeout = timeout_tt

    # ========== STEP 2: PATH PLANNING SYSTEM ==========

    def calculate_waypoints(self, start_pose, end_pose, operation_type="generic"):
        """Generate collision-free waypoint path

        Args:
            start_pose: [x, y, z, rx, ry, rz] current pose
            end_pose: [x, y, z, rx, ry, rz] target pose
            operation_type: "push", "grip", "place", "generic"

        Returns:
            List of waypoint poses including start and end
        """
        waypoints = []
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])

        # Calculate tower top for safe height reference
        tower_top = self.get_tower_top_height()
        safe_z = tower_top + self.SAFE_HEIGHT

        # Calculate distance from tower center (in XY plane)
        start_dist = np.linalg.norm(start_pos[:2] - self.TOWER_CENTER[:2])
        end_dist = np.linalg.norm(end_pos[:2] - self.TOWER_CENTER[:2])

        # If both start and end are close to tower, use 3-phase path
        if start_dist < self.TOWER_CLEARANCE_RADIUS or end_dist < self.TOWER_CLEARANCE_RADIUS:
            # Phase 1: Retract - lift to safe height while maintaining XY
            retract_pose = start_pose.copy()
            retract_pose[2] = max(safe_z, start_pose[2] + self.RETRACT_OFFSET)
            waypoints.append(retract_pose)

            # Phase 2: Transit - move to target XY at safe height
            transit_pose = end_pose.copy()
            transit_pose[2] = safe_z
            waypoints.append(transit_pose)

            # Phase 3: Approach - descend to target
            waypoints.append(end_pose)
        else:
            # Simple direct path if far from tower
            waypoints.append(end_pose)

        return waypoints

    def is_near_tower(self, position):
        """Check if a position is within the tower clearance zone

        Args:
            position: [x, y, z] or numpy array

        Returns:
            bool: True if within tower clearance radius
        """
        pos = np.array(position[:3])
        dist_xy = np.linalg.norm(pos[:2] - self.TOWER_CENTER[:2])
        return dist_xy < self.TOWER_CLEARANCE_RADIUS

    # ========== STEP 3: BLOCK SELECTION LOGIC ==========

    def select_next_block(self):
        """Select the next safe block to remove

        Strategy:
        - Avoid layer 1 (bottom) and layer 9 (top, reserved for placement)
        - Prefer middle blocks (block 2) from each layer
        - Move from lower to higher layers
        - Don't take from same layer consecutively (leave at least 1 block per layer)

        Returns:
            (layer, block_num) tuple or None if no safe blocks available
        """
        # Priority order: layers 2-8, prefer middle blocks
        for layer in range(2, 9):  # Layers 2 through 8
            # Skip if this layer already has 2 blocks removed
            removed_from_layer = sum(1 for (l, b) in self.removed_blocks if l == layer)
            if removed_from_layer >= 2:
                continue

            # Try middle block first (block 2), then sides (1, 3)
            for block_num in [2, 1, 3]:
                if (layer, block_num) not in self.removed_blocks:
                    # Found a valid block
                    return (layer, block_num)

        # No safe blocks found
        return None

    def mark_block_removed(self, layer, block_num):
        """Mark a block as removed from the tower

        Args:
            layer: Layer number
            block_num: Block number in layer
        """
        self.removed_blocks.add((layer, block_num))
        self.touched_layers.add(layer)

    def is_block_removed(self, layer, block_num):
        """Check if a block has been removed

        Args:
            layer: Layer number
            block_num: Block number in layer

        Returns:
            bool: True if block was removed
        """
        return (layer, block_num) in self.removed_blocks

    # ========== STEP 4.1: POSE CALCULATION FUNCTIONS ==========

    def calculate_push_pose(self, block_pos, layer):
        """Calculate pose for pushing block

        Args:
            block_pos: [x, y, z] of block center (numpy array or list)
            layer: Layer number (for orientation)

        Returns:
            [x, y, z, rx, ry, rz] pose for pushing
        """
        pos = np.array(block_pos)
        is_odd_layer = (layer % 2) == 1

        if is_odd_layer:
            # Odd layers: blocks parallel to X-axis
            # Push from -Y side (robot side)
            push_pos = pos.copy()
            push_pos[1] -= (self.BLOCK_WIDTH / 2 + self.APPROACH_OFFSET)
            # Gripper points in +Y direction (fingers along X)
            orientation = [0, 0, 0]  # Gripper pointing up, fingers along X
        else:
            # Even layers: blocks parallel to Y-axis
            # Push from -X side
            push_pos = pos.copy()
            push_pos[0] -= (self.BLOCK_WIDTH / 2 + self.APPROACH_OFFSET)
            # Gripper points in +X direction (fingers along Y)
            orientation = [0, 0, math.pi/2]  # Rotated 90° around Z

        return list(push_pos) + orientation

    def calculate_grip_pose(self, block_pos, layer, extended=True):
        """Calculate pose for gripping block

        Args:
            block_pos: [x, y, z] of block center
            layer: Layer number
            extended: True if block has been pushed out

        Returns:
            [x, y, z, rx, ry, rz] pose for gripping
        """
        pos = np.array(block_pos)
        is_odd_layer = (layer % 2) == 1

        if is_odd_layer:
            # Odd layers: blocks parallel to X-axis
            # Grip from +Y side (opposite from push)
            grip_pos = pos.copy()
            if extended:
                # Block has been pushed toward +Y, so approach from +Y
                grip_pos[1] += (self.PUSH_DEPTH + self.BLOCK_WIDTH / 2)
            else:
                grip_pos[1] += (self.BLOCK_WIDTH / 2 + self.APPROACH_OFFSET)
            # Gripper points in -Y direction
            orientation = [0, 0, math.pi]  # Rotated 180° to point back
        else:
            # Even layers: blocks parallel to Y-axis
            # Grip from +X side (opposite from push)
            grip_pos = pos.copy()
            if extended:
                # Block has been pushed toward +X
                grip_pos[0] += (self.PUSH_DEPTH + self.BLOCK_WIDTH / 2)
            else:
                grip_pos[0] += (self.BLOCK_WIDTH / 2 + self.APPROACH_OFFSET)
            # Gripper points in -X direction
            orientation = [0, 0, -math.pi/2]  # Rotated to point back

        return list(grip_pos) + orientation

    def calculate_place_pose(self, tower_height, block_num):
        """Calculate pose for placing block on top

        Args:
            tower_height: Current Z-height of tower top
            block_num: Which block in the new layer (1-3, left to right)

        Returns:
            [x, y, z, rx, ry, rz] pose for placement
        """
        # Determine orientation of new layer based on blocks already placed
        # Layer 9 is odd (parallel to X), layer 10 would be even, etc.
        current_top_layer = 9 + (self.blocks_placed_on_top // 3)
        is_odd_layer = (current_top_layer % 2) == 1

        # Calculate position for this block in the new layer
        place_pos = self.TOWER_CENTER.copy()
        place_pos[2] = tower_height + self.BLOCK_HEIGHT / 2

        if is_odd_layer:
            # Blocks parallel to X-axis: vary in X direction
            offset = (block_num - 2) * self.BLOCK_SPACING  # -1, 0, +1 spacing
            place_pos[0] += offset
            orientation = [0, 0, 0]  # Gripper aligned with X
        else:
            # Blocks parallel to Y-axis: vary in Y direction
            offset = (block_num - 2) * self.BLOCK_SPACING
            place_pos[1] += offset
            orientation = [0, 0, math.pi/2]  # Rotated 90°

        return list(place_pos) + orientation

    # ========== STEP 4.2: MOTION CONTROL FUNCTIONS ==========

    def move_to_pose(self, target_pose, current_q):
        """Calculate joint angles for target pose using IK

        Args:
            target_pose: [x, y, z, rx, ry, rz]
            current_q: Current joint angles (6 elements)

        Returns:
            Target joint angles (6 elements) or None if IK fails
        """
        try:
            target_q = inverseKinematics(target_pose, current_q)
            return target_q
        except Exception as e:
            print(f"IK failed for pose {target_pose}: {e}")
            return None

    def is_motion_complete(self):
        """Check if current motion is complete

        Returns:
            bool: True if not moving
        """
        return not self.is_moving

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
        print(current_q)

        # pose = center left of tower, raised up 0.2 on Z with end rotated pi/2 around X
        target_pose = np.append(
            self.TOWER_CENTER - self.ROBOT_BASE + [0, self.BLOCK_WIDTH * 1.5 + 0.25, 0],
            [-1.2092, -1.2092, 1.2092],
        )
        target_pose[2] += 0.2
        target_q = self.move_to_pose(target_pose, cur_angles)
        if target_q is not None:
            self.set_target(np.array(list(target_q) + [False]), cur_angles)

        # done or error
        gripper_state = (
            self.target_q[6]
            if self.target_q is not None and len(self.target_q) >= 7
            else False
        )
        return np.append(current_q[:6], gripper_state)
