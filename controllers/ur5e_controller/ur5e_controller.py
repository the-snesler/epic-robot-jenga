"""ur5e_controller controller."""

from controller import Robot, Supervisor
import numpy as np

# ===== Optional: fast live preview with OpenCV =====
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    print("[camera] OpenCV not available; will skip live preview.")

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# -----------------------------------------------------------------------------
# UR5e arm joints (6-DoF)
# -----------------------------------------------------------------------------
arm_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
motors = [robot.getDevice(n) for n in arm_joint_names]
for m in motors:
    ps = m.getPositionSensor()
    ps.enable(timestep)

# -----------------------------------------------------------------------------
# Robotiq 2F-85 gripper (two finger motors under toolSlot)
# Names come from your world: Robotiq2f85Gripper { name "robotiq_2f85" }
# So the device names are:
#   "robotiq_2f85::left finger joint", "robotiq_2f85::right finger joint"
# -----------------------------------------------------------------------------
GRIP_MIN = 0.0
GRIP_MAX = 0.80  # typical range on actuated finger joints; adjust if needed

def _safe_get_device(name):
    try:
        return robot.getDevice(name)
    except Exception:
        return None

gripper_left  = _safe_get_device("robotiq_2f85::left finger joint")
gripper_right = _safe_get_device("robotiq_2f85::right finger joint")

if gripper_left and gripper_right:
    for g in (gripper_left, gripper_right):
        try:
            g.setVelocity(1.5)
        except Exception:
            pass
        try:
            g.setForce(50.0)
        except Exception:
            pass
    print("[gripper] Robotiq 2F-85 finger motors found.")
else:
    print("[gripper] Could not find Robotiq 2F-85 motors. "
          "Expected: 'robotiq_2f85::left finger joint' and "
          "'robotiq_2f85::right finger joint'.")

def set_gripper_q(q):
    """Set finger joint target (radians) to both fingers."""
    if not (gripper_left and gripper_right):
        return
    q = float(np.clip(q, GRIP_MIN, GRIP_MAX))
    gripper_left.setPosition(q)
    gripper_right.setPosition(q)

def set_gripper_normalized(g):
    """g in [0,1]: 0=open, 1=closed."""
    g = float(np.clip(g, 0.0, 1.0))
    q = GRIP_MIN + g * (GRIP_MAX - GRIP_MIN)
    set_gripper_q(q)

def set_gripper_width_mm(width_mm, max_open_mm=85.0):
    """Approximate map from opening width (mm) to joint value; tune as needed."""
    width_mm = float(np.clip(width_mm, 0.0, max_open_mm))
    set_gripper_normalized(1.0 - width_mm / max_open_mm)

# -----------------------------------------------------------------------------
# Wrist camera (correct BGRA → BGR for OpenCV preview)
# -----------------------------------------------------------------------------
cam = _safe_get_device("wrist_camera")
if cam:
    cam.enable(timestep)
    cam_w, cam_h = cam.getWidth(), cam.getHeight()
    print(f"[camera] Enabled wrist_camera at {cam_w}x{cam_h}")
else:
    cam_w = cam_h = None
    print("[camera] No device named 'wrist_camera' found.")

def _camera_step_preview():
    if not (cam and _HAS_CV2):
        return
    img_bytes = cam.getImage()
    if img_bytes is None:
        return
    # Webots returns BGRA
    bgra = np.frombuffer(img_bytes, dtype=np.uint8).reshape((cam_h, cam_w, 4))
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Wrist Camera", bgr)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC closes
        cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
tt = 0
try:
    while robot.step(timestep) != -1:
        # Example UR5e pose (replace with your planner)
        desired_arm = [-0.343, -1.2, 1.5, -2.0, -1.57, 1.03]
        for j, motor in enumerate(motors):
            motor.setPosition(desired_arm[j])

        # Demo gripper motion: close over 2s, hold to 4s, then open
        t = robot.getTime()
        if t < 2.0:
            set_gripper_normalized(t / 2.0)   # ramp 0→1
        elif t < 4.0:
            set_gripper_normalized(1.0)       # closed
        else:
            set_gripper_width_mm(85.0)        # fully open

        _camera_step_preview()
        tt += 1
finally:
    if _HAS_CV2:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
