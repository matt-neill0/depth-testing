#!/usr/bin/env python3
"""
Real-time stereo disparity & depth viewer using NVIDIA VPI.

Works with either:
  • Intel RealSense D435/D455           (stereo IR images auto-detected)
  • Two generic USB/RGB cameras         (--source usb  --left_id … --right_id …)

If you provide focal length (fx) and baseline for USB cameras, the script
converts disparity → metric depth; otherwise it shows disparity only.

Author: Matt Neill
"""

# ────────────────── stdlib / third-party imports ──────────────────
import argparse, threading, time
import cv2, numpy as np, vpi                 # VPI 3.x Python bindings
import CameraCapture as cam                  # helper for grabbing frames

# ────────────────────── command-line arguments ───────────────────
parser = argparse.ArgumentParser(description="VPI real-time stereo viewer")

parser.add_argument("--source",   choices=["realsense", "usb"],
                    default="realsense",
                    help="Camera source (default: realsense)")

# USB-camera specific IDs
parser.add_argument("--left_id",  type=int, default=0,
                    help="Left USB cam index (usb mode)")
parser.add_argument("--right_id", type=int, default=1,
                    help="Right USB cam index (usb mode)")

# Optional calibration for USB cameras
parser.add_argument("--fx",       type=float,
                    help="Focal length in pixels (usb mode)")
parser.add_argument("--baseline", type=float,
                    help="Baseline in metres (usb mode)")

parser.add_argument("--max_disp", type=int, default=128,
                    choices=[64, 128, 256],
                    help="Maximum disparity VPI will search (default 256)")

args = parser.parse_args()

# ────────────────── helper: build vertical colour-bar legend ──────────────────
def make_colourbar(colourmap,
                   meters_min: float, meters_max: float,
                   height: int = 256,
                   ramp_width: int = 40,
                   pad_lr: int = 70,
                   pad_tb: int = 8,
                   n_ticks: int = 5) -> np.ndarray:
    """
    Returns an (H, W, 3) BGR image:
      * left pad = rainbow ramp (blue near, red far)
      * right pad = metre labels
    """
    full_h = height + 2 * pad_tb

    # gradient 255 → 0  (255 becomes red with COLORMAP_JET)
    grad = np.linspace(255, 0, height, dtype=np.uint8).reshape(-1, 1)
    grad = np.repeat(grad, ramp_width, axis=1)
    ramp_bgr = cv2.applyColorMap(grad, colourmap)

    # canvas: ramp + label area
    bar_bgr = np.zeros((full_h, ramp_width + pad_lr, 3), np.uint8)
    bar_bgr[pad_tb:pad_tb + height, :ramp_width] = ramp_bgr

    # draw n_ticks labels evenly spaced, top = far (meters_max)
    step, x_txt = height // (n_ticks - 1), ramp_width + 6
    for i in range(n_ticks):
        y = pad_tb + i * step
        depth_m = meters_max - (meters_max - meters_min) * (i / (n_ticks - 1))
        label   = f"{depth_m:0.1f} m"
        # black outline + white foreground for readability
        cv2.putText(bar_bgr, label, (x_txt, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar_bgr, label, (x_txt, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bar_bgr

# ───────────────────────── global parameters ─────────────────────────
MAX_DISP  = args.max_disp
SCALE_U8  = 255.0 / (32 * MAX_DISP)          # fixed-point Q10.5 -> byte
COLOURMAP = cv2.COLORMAP_JET                 # blue-green-red ramp
CLIP_MIN, CLIP_MAX = 0, 4                    # depth clip for visualisation (m)

# FPS tracking (exponential moving average)
prev_t, fps, ALPHA = time.time(), 0.0, 0.9

# ─────────────────────── start camera threads ───────────────────────
if args.source == "realsense":
    threading.Thread(target=cam.capture_realsense, daemon=True).start()
    fx_px, baseline_m = cam.get_calibration()    # blocking until ready
    depth_enabled = True
    print(f"[Init] RealSense fx={fx_px:.1f}px  b={baseline_m*1e3:.1f} mm")
else:
    # spawn two UVC / v4l2 threads
    threading.Thread(target=cam.capture_camera,
                     args=(args.left_id, True), daemon=True).start()
    threading.Thread(target=cam.capture_camera,
                     args=(args.right_id, False), daemon=True).start()

    if args.fx and args.baseline:
        fx_px, baseline_m = args.fx, args.baseline
        depth_enabled = True
        print(f"[Init] USB stereo fx={fx_px:.1f}px  b={baseline_m*1e3:.1f} mm")
    else:
        depth_enabled = False
        print("[Init] USB stereo without calibration → depth disabled")

# ───────── pre-allocate GPU images (lazy, resized on first frame) ────────
disp_s16_gpu = disp_u8_gpu = conf_u16_gpu = None
stop = False

# ───────────────────────── main loop ─────────────────────────
while not stop:
    left, right = cam.get_frames()   # (frame1, frame2) as uint8 NumPy
    if left is None:
        time.sleep(0.005)            # wait until first frames arrive
        continue

    # ─── FPS update ───
    now = time.time()
    fps  = ALPHA * fps + (1 - ALPHA) / (now - prev_t)
    prev_t = now
    fps_txt = f"{fps:5.1f} FPS"

    cv2.imshow("Left camera",  left)
    cv2.imshow("Right camera", right)

    # ─── disparity computation with VPI ───
    try:
        with vpi.Backend.CUDA:
            left_gpu  = vpi.asimage(left,  vpi.Format.U8)
            right_gpu = vpi.asimage(right, vpi.Format.U8)

            # recreate GPU buffers if resolution changed
            if conf_u16_gpu is None or conf_u16_gpu.size != left_gpu.size:
                disp_s16_gpu = vpi.Image(left_gpu.size, vpi.Format.S16)
                disp_u8_gpu  = vpi.Image(left_gpu.size, vpi.Format.U8)
                conf_u16_gpu = vpi.Image(left_gpu.size, vpi.Format.U16)

            # run block-matching stereo with stronger smoothness & confidence
            disp_s16 = vpi.stereodisp(left_gpu, right_gpu,
                                      maxdisp       = MAX_DISP,
                                      window        = 9,
                                      p1            = 8,
                                      p2            = 96,
                                      uniqueness    = 0.9,
                                      includediagonals=False,
                                      out_confmap   = conf_u16_gpu)

            # convert disparity to pitch-linear S16, scale to U8, confidence→U8
            disp_s16.convert(disp_s16_gpu)
            conf_u8_gpu = conf_u16_gpu.convert(vpi.Format.U8,
                                               scale=255.0/65535)
            disp_s16_gpu.convert(disp_u8_gpu, scale=SCALE_U8)

        # ─── post-filter disparity ───
        disp16 = disp_s16_gpu.cpu().copy()      # signed 16-bit, Q10.5
        conf_u8 = conf_u8_gpu.cpu()             # 0-255 confidence
        disp16[conf_u8 < 30] = -1               # mask low-conf pixels

        disp_u8 = cv2.normalize(disp16, None, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # colourise & overlay FPS
        disp_col  = cv2.applyColorMap(disp_u8, COLOURMAP)
        disp_show = disp_col.copy()
        cv2.putText(disp_show, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(disp_show, fps_txt, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("Disparity", disp_show)

        # ─── depth map (requires fx & baseline) ───
        if depth_enabled:
            disp_px = disp16.astype(np.float32) / 32.0      # Q10.5 → pixels
            depth_m = np.full_like(disp_px, np.nan, np.float32)
            depth_m[disp_px > 0] = (fx_px * baseline_m) / disp_px[disp_px > 0]

            # clip for colour-map visual
            depth_m[depth_m < CLIP_MIN] = CLIP_MIN
            depth_m[depth_m > CLIP_MAX] = CLIP_MAX
            depth_m[np.isnan(depth_m)]  = CLIP_MIN

            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
            depth_col = cv2.applyColorMap(depth_vis.astype(np.uint8), COLOURMAP)

            # build legend & stack beside depth image
            legend = make_colourbar(COLOURMAP, CLIP_MIN, CLIP_MAX,
                                    height=left.shape[0])
            if legend.shape[0] != depth_col.shape[0]:
                legend = cv2.resize(legend,
                                    (legend.shape[1], depth_col.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            depth_with_leg = np.hstack((depth_col, legend))

            depth_show = depth_with_leg.copy()
            cv2.putText(depth_show, fps_txt, (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(depth_show, fps_txt, (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Depth", depth_show)

    except Exception as err:
        print("[Loop] VPI error:", err)

    # graceful exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        cam.signal_stop()

# clean shutdown
cv2.destroyAllWindows()
print("[Exit] Done.")
