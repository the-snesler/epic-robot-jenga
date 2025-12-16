import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from controller.node import Node

def T01(q):
    c1, s1 = np.cos(q), np.sin(q)
    return np.array([
        [ c1,  0.0, -s1,  0.0],
        [ s1,  0.0,  c1,  0.0],
        [ 0.0, -1.0, 0.0, 0.1625],
        [ 0.0,  0.0, 0.0,  1.0]
    ])

def T12(q):
    c2, s2 = np.cos(q), np.sin(q)
    return np.array([
        [ c2, -s2, 0.0, 0.4250*c2],
        [ s2,  c2, 0.0, 0.4250*s2],
        [ 0.0, 0.0, 1.0,  0.0],
        [ 0.0, 0.0, 0.0,  1.0]
    ])

def T23(q):
    c3, s3 = np.cos(q), np.sin(q)
    return np.array([
        [ c3, -s3, 0.0, 0.3922*c3],
        [ s3,  c3, 0.0, 0.3922*s3],
        [ 0.0, 0.0, 1.0,  0.0],
        [ 0.0, 0.0, 0.0,  1.0]
    ])

def T34(q):
    c4, s4 = np.cos(q), np.sin(q)
    return np.array([
        [ c4,  0.0, -s4,  0.0],
        [ s4,  0.0,  c4,  0.0],
        [ 0.0, -1.0, 0.0, 0.1333],
        [ 0.0,  0.0, 0.0,  1.0]
    ])

def T45(q):
    c5, s5 = np.cos(q), np.sin(q)
    return np.array([
        [ c5,  0.0,  s5,  0.0],
        [ s5,  0.0, -c5,  0.0],
        [ 0.0,  1.0,  0.0, 0.0997],
        [ 0.0,  0.0,  0.0,  1.0]
    ])

def T56(q):
    c6, s6 = np.cos(q), np.sin(q)
    return np.array([
        [ c6,  +s6,  0.0,  0.0],   
        [ s6,  -c6,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0996], 
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    
def forwardKinematics(q):
    """Computes the forward kinematics for the UR5e robot

    Args:
        q: a list of six joint angles in radians
    Returns:
        A 4x4 numpy array representing the homogeneous transformation matrix of the end-effector with respect to the base frame
    """
    T = T01(q[0]) @ T12(q[1]) @ T23(q[2]) @ T34(q[3]) @ T45(q[4]) @ T56(q[5])
    return T

def transformMatrixToPose(T):
    """Converts a homogeneous transformation matrix to a pose [x,y,z,wx,wy,wz] where w is in exponential coordinates.

    Args:
        T: A 4x4 numpy array representing the homogeneous transformation matrix
    Returns:
        A six-element numpy array representing the position (3) and orientation (3, exponential coordinates)
    """
    p = T[:3, 3]
    Rm = T[:3, :3]
    w = R.from_matrix(Rm).as_rotvec()
    return np.concatenate([p, w])

def close_enough(a, b, tol=0.02):
    return np.linalg.norm(np.array(a) - np.array(b)) < tol

def detect_object(image_bgr, color) -> tuple[float, float] | None:
    """Detects the angle and distance between the center of the image and the center of the largest object of the specified color in radians"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lower_bound = np.array([max(0, c - 70) for c in color])
    upper_bound = np.array([min(255, c + 70) for c in color])
    mask = cv2.inRange(rgb, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    img_center_x = image_bgr.shape[1] // 2 + 57 # adjust for camera offset 
    img_center_y = image_bgr.shape[0] // 2
    delta_x = cx - img_center_x
    delta_y = img_center_y - cy
    angle = math.atan2(delta_y, delta_x)
    distance = math.hypot(delta_x, delta_y)
    return angle, distance
    
def getFK(q):
    return T01(q[0]) @ T12(q[1]) @ T23(q[2]) @ T34(q[3]) @ T45(q[4]) @ T56(q[5])
def getJacobian(q):
    J = np.zeros((6, 6))
    T06 = getFK(q)
    p_e = T06[:3, 3]
    # Prefix transforms: include identity so the first joint's axis is in base
    T_list = [np.eye(4)]
    T_list.append(T01(q[0]))
    T_list.append(T_list[1] @ T12(q[1]))
    T_list.append(T_list[2] @ T23(q[2]))
    T_list.append(T_list[3] @ T34(q[3]))
    T_list.append(T_list[4] @ T45(q[4]))
    T_list.append(T_list[5] @ T56(q[5]))
    for i in range(6):
        Ti = T_list[i]
        Ri = Ti[:3, :3]
        pi = Ti[:3, 3]
        # Joint axis z in the local frame → in base
        if i == 0:
            zi = np.array([0.0, 0.0, 1.0])
        else:
            zi = Ri @ np.array([0.0, 0.0, 1.0])
        J[:3, i] = np.cross(zi, (p_e - pi))
        J[3:, i] = zi
    return J
def poseError(current_pose, desired_pose):
    cp = current_pose[:3]
    dp = desired_pose[:3]
    pos_err = dp - cp
    c_exp = current_pose[3:]
    d_exp = desired_pose[3:]
    Rc = R.from_rotvec(c_exp).as_matrix()
    Rd = R.from_rotvec(d_exp).as_matrix()
    Rrel = Rd @ Rc.T
    ori_err = R.from_matrix(Rrel).as_rotvec()
    return np.concatenate([pos_err, ori_err])
def getIK(desired_pose, current_q):
    """
    Newton–Raphson IK with damped pseudo-inverse (via SVD pinv).
    desired_pose: [x,y,z, wx,wy,wz] (position + exponential coordinates).
    Returns the joint vector that best reaches desired_pose from current_q.
    """
    q = np.array(current_q, dtype=float)
    desiredPose = np.array(desired_pose, dtype=float)
    maxIterations = 200 # Can be changed
    tolerance = 1e-8 # Can be changed
    alpha = 0.03 # Can be changed
   
    for _ in range(maxIterations):
        T06 = getFK(q)
        currentPosition = T06[:3, 3]
        currentRotationMatrix = T06[:3, :3]
        currentExp = R.from_matrix(currentRotationMatrix).as_rotvec()
        currentPose = np.concatenate([currentPosition, currentExp])
       
        error = poseError(currentPose, desiredPose)
       
        if np.linalg.norm(error) < tolerance:
            break
       
        jacobian = getJacobian(q)
       
        jacobianInv = np.linalg.pinv(jacobian)
       
        deltaQ = alpha * jacobianInv @ error
        q += deltaQ
   
    return q.tolist()
    
def get_block_position(root: Node, layer, block_num):
    """
    Retrieve the world position of a block from Webots.

    Args:
        layer: Layer index (1–9)
        block_num: Block index (1–3)

    Returns:
        np.array([x, y, z]) or None if not found
    """

    block_name = f"solid({layer}{block_num})"

    try:
        children_field = root.getField("children")

        # Search entire scene tree recursively
        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)
            if node is None:
                continue

            translation_field = _find_translation_by_name(node, block_name)
            if translation_field is not None:
                return translation_field

        print(f"Warning: Block {block_name} not found")
        return None

    except Exception as e:
        print(f"Error retrieving block {block_name}: {e}")
        return None


def get_block_node(root: Node, layer, block_num) -> Node | None:
    """
    Retrieve the Webots node reference for a block.

    Args:
        root: Root node from supervisor
        layer: Layer index (1–9)
        block_num: Block index (1–3)

    Returns:
        Webots Node object or None if not found
    """
    block_name = f"solid({layer}{block_num})"

    try:
        children_field = root.getField("children")

        # Search entire scene tree recursively
        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)
            if node is None:
                continue

            found_node = _find_node_by_name(node, block_name)
            if found_node is not None:
                return found_node

        print(f"Warning: Block node {block_name} not found")
        return None

    except Exception as e:
        print(f"Error retrieving block node {block_name}: {e}")
        return None


def _find_translation_by_name(node, target_name):
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
            translation_field = node.getField("translation")
            if translation_field:
                return np.array(translation_field.getSFVec3f())

    # Recursively search children
    translation_field = node.getField("translation")
    try:
        children_field = node.getField("children")
        if children_field:
            for i in range(children_field.getCount()):
                child = children_field.getMFNode(i)
                result = _find_translation_by_name(child, target_name)
                if result is not None and translation_field:
                    return np.array(translation_field.getSFVec3f()) + result
                elif result is not None:
                    return result
    except:
        pass

    return None


def _find_node_by_name(node, target_name):
    """
    Recursively search for a node by its internal name and return the node itself.

    Args:
        node: Node to search
        target_name: Name string to match

    Returns:
        Webots Node object or None
    """
    if node is None:
        return None

    # Check name field
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
                result = _find_node_by_name(child, target_name)
                if result is not None:
                    return result
    except:
        pass

    return None
