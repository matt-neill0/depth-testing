"""
Runs Depth-Anything V2 on **one** webcam stream and show the
relative-depth map next to the live image.

– Grabs frames with the CameraCapture helper (any UVC camera).
– Converts the gray frame to RGB (Depth-Anything expects colour).
– Uses Hugging-Face `pipeline` for a single-line forward pass.
"""

from transformers import pipeline               # HF task pipeline wrapper
import CameraCapture as cam                     # your frame-grab helper
import threading, cv2, numpy as np
from PIL import Image                           # pipeline accepts PIL.Image

# ─────────────────── start webcam thread ────────────────────
# capture_camera(cam_id=4, is_left=True) → stores frames in cam._frame1
camera = threading.Thread(target=cam.capture_camera,
                          args=(4, True), daemon=True)
camera.start()

# ─────────────────── Depth-Anything pipeline ────────────────
#   • model:  V2-Large (highest fidelity, slowest)
#   • device: "cuda:0"  — change to "cpu" if no GPU available
pipe = pipeline("depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device="cuda:0")

COLOURMAP = cv2.COLORMAP_TURBO

stop = False
while not stop:
    left, _ = cam.get_frames()      # ignore right-eye frame, Depth-Anything is monocular
    if left is None:
        continue

    # OpenCV BGR/Gray → RGB → PIL.Image for the HF pipeline
    rgb_np  = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    rgb_pil = Image.fromarray(rgb_np)

    # forward pass → dict with a PIL depth image ("depth" key)
    depth_pil = pipe(rgb_pil)["depth"]
    depth_np  = np.array(depth_pil)      # 8-bit relative depth for imshow

    depth_col = cv2.applyColorMap(depth_np, COLOURMAP)

    # ───── display ─────
    cv2.imshow("Camera (gray)", left)
    cv2.imshow("DepthAnything relative depth", depth_col)

    # press Q → clean shutdown
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        cam.signal_stop()

cv2.destroyAllWindows()
