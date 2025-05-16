import cv2, threading, time, numpy as np, pyrealsense2 as rs
# This module is imported by depth testing .py files.  It provides two “grabber” threads:
#   • capture_camera   – for generic UVC/-v4l2 webcams
#   • capture_realsense – for Intel RealSense D435/D455 (left/right IR)
# plus thread-safe helpers to fetch frames and calibration.

# ───────────────────────── shared state ─────────────────────────
_lock       = threading.Lock()     # guards every read/write to _frame1/_frame2
_frame1     = None                 # latest LEFT  frame (NumPy uint8, Gray)
_frame2     = None                 # latest RIGHT frame (NumPy uint8, Gray)
_stop_flag  = False                # set → all threads exit cleanly
_ready      = threading.Event()    # set once fx + baseline are known

# stereo intrinsics (populated exactly once by RealSense thread)
_fx         = None                 # focal length in **pixels**
_baseline_m = None                 # baseline in **metres**

# ───────────────────── generic USB-/webcam thread ────────────────────
def capture_camera(cam_id: int, is_left: bool):
    """
    Grab grayscale frames from a USB/RGB camera and store them in _frame1/2.
    Runs forever until signal_stop() is called or the camera stalls.
    """
    global _frame1, _frame2, _stop_flag
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[CameraCapture] Cannot open camera {cam_id}")
        _stop_flag = True
        return

    while not _stop_flag:
        ok, frame = cap.read()
        if not ok:
            print(f"[CameraCapture] Lost camera {cam_id}")
            _stop_flag = True
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with _lock:
            if is_left:
                _frame1 = gray
            else:
                _frame2 = gray
    cap.release()

# ──────────────────────── RealSense helper ──────────────────────────
def _rs_device_available(timeout_s: int = 2) -> bool:
    """Return True if any RealSense enumerates on USB-3 within timeout."""
    ctx, t0 = rs.context(), time.time()
    while time.time() - t0 < timeout_s:
        if any(dev.supports(rs.camera_info.product_line) for dev in ctx.query_devices()):
            return True
        time.sleep(0.1)
    return False

# ─────────────────────── RealSense stereo thread ────────────────────
def capture_realsense():
    """
    Continuously grab Infra 1 (left) and Infra 2 (right) grayscale frames
    from a D435/D455.  Automatically retries on cable/USB drops.
    Populates global _fx and _baseline_m once, then sets _ready.
    """
    global _frame1, _frame2, _fx, _baseline_m, _stop_flag

    if not _rs_device_available():
        print("[CameraCapture] No RealSense found on USB-3 → giving up")
        _stop_flag = True
        return

    while not _stop_flag:
        # camera may disappear, so we create a new pipeline each retry loop
        pipeline, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 30)

        try:
            prof = pipeline.start(cfg)

            # one-time fetch of fx + baseline for depth conversion
            if _fx is None:
                intr = prof.get_stream(rs.stream.infrared,1)\
                           .as_video_stream_profile().get_intrinsics()
                extr = prof.get_stream(rs.stream.infrared,1)\
                           .get_extrinsics_to(prof.get_stream(rs.stream.infrared,2))
                _fx         = intr.fx
                _baseline_m = abs(extr.translation[0])
                _ready.set()                        # unblock get_calibration()

            # main frame-grab loop
            while not _stop_flag:
                if not pipeline.poll_for_frames():  # non-blocking USB check
                    time.sleep(0.002)
                    continue
                frames = pipeline.wait_for_frames(timeout_ms=100)
                ir_l   = frames.get_infrared_frame(1)
                ir_r   = frames.get_infrared_frame(2)
                if not ir_l or not ir_r:            # rare sync miss
                    continue
                with _lock:
                    _frame1 = np.asanyarray(ir_l.get_data())
                    _frame2 = np.asanyarray(ir_r.get_data())

        except Exception as e:                      # USB glitch, etc.
            print(f"[CameraCapture] RealSense error → {e}")
            pipeline.stop()
            time.sleep(0.5)                         # back-off then retry
        else:
            pipeline.stop()                         # clean exit (stop_flag)

# ───────────────────────── helper getters ───────────────────────
def get_frames():
    """
    Thread-safe access to the latest stereo pair.
    Returns (left,right) *copies* so callers can modify freely.
    """
    with _lock:
        return (_frame1.copy() if _frame1 is not None else None,
                _frame2.copy() if _frame2 is not None else None)

def get_calibration():
    """Block until _ready is set, then return (fx_px, baseline_m)."""
    _ready.wait()
    return _fx, _baseline_m

def signal_stop():
    """Ask all capture threads to terminate gracefully."""
    global _stop_flag
    _stop_flag = True
