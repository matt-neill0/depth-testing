# ─────────────── imports ───────────────
import cv2, time, torch, threading, numpy as np
from torchvision.transforms.functional import to_tensor
import CameraCapture as cam
from PIL import  Image

# ─────────── user-configurable flags ───────────
CAM_ID = 4
RUN_ON_GPU = True
IMG_SIZE    = (384, 768)

# ──────────── load ZoeDepth model ─────────────
device = torch.device("cuda:0" if RUN_ON_GPU and torch.cuda.is_available() else "cpu")

print("[Init] Loading ZoeDepth from PyTorch Hub...")
model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
model = model.to(device).eval()
print("[Init] ZoeDepth ready on", device)

COLOURMAP = cv2.COLORMAP_TURBO

# ──────────────── FPS helper ───────────────
prev_t, fps, ALPHA = time.time(), 0.0, 0.9

# ──────────────── open camera ──────────────
camera = threading.Thread(target=cam.capture_camera,
                          args=(4, True), daemon=True)
camera.start()

# ────────────────── main loop ─────────────────
with torch.no_grad():
    while True:
        frame, _ = cam.get_frames()  # ignore right-eye frame
        if frame is None:
            continue

        # ─── FPS update ───
        now = time.time()
        fps  = ALPHA*fps + (1-ALPHA)/(now-prev_t)
        prev_t = now
        fps_txt = f"{fps:5.1f} FPS"

        # ─── preprocess for ZoeDepth ───
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        img_tensor = to_tensor(frame_rgb)[None].to(device)      # 1×3×H×W, 0-1

        # ─── inference ───
        depth_pred = model.infer(img_tensor)   # 1×1×H×W, float32
        depth_np   = depth_pred.squeeze().cpu().numpy()

        # normalise to 0-255 for display (relative depth)
        depth_vis = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_col = cv2.applyColorMap(depth_vis, COLOURMAP)

        # ─── overlay FPS on both windows ───
        cv2.putText(frame, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(depth_col, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(depth_col, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        # ─── show ───
        cv2.imshow("Camera", frame)
        cv2.imshow("ZoeDepth relative depth", depth_col)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.signal_stop()
            break

cv2.destroyAllWindows()