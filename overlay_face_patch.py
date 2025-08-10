#!/usr/bin/env python3
import argparse, math, os
import cv2
import numpy as np

# --- add these imports at the top ---
import sys
try:
    import mediapipe as mp
    HAS_MP = True
except Exception:
    HAS_MP = False

# --- robust detector ---
def detect_face_any(gray, min_face=80, aggressive=False, try_mediapipe=True, boost=1.5, use_clahe=False):
    H, W = gray.shape[:2]
    # 0) optional pre-boost for blurry input
    g = cv2.resize(gray, None, fx=boost, fy=boost, interpolation=cv2.INTER_CUBIC) if boost and boost != 1.0 else gray.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    else:
        g = cv2.equalizeHist(g)

    # 1) Haar (normal/aggressive)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # normal
    params = dict(scaleFactor=1.2, minNeighbors=5, minSize=(min_face, min_face))
    faces = cascade.detectMultiScale(g, **params)
    if len(faces) == 0 and aggressive:
        # aggressive: smaller scale step, fewer neighbors, smaller min size
        params = dict(scaleFactor=1.05, minNeighbors=2, minSize=(int(min_face*0.6), int(min_face*0.6)))
        faces = cascade.detectMultiScale(g, **params)

    if len(faces):
        # convert back to original coords (we upscaled 1.5x)
        x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
        x = int(x/boost); y = int(y/boost); w = int(w/boost); h = int(h/boost)
        return (x,y,w,h)

    # 2) MediaPipe fallback (more robust to blur/low light)
    if try_mediapipe and HAS_MP:
        mpfd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        res = mpfd.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        if res.detections:
            det = max(res.detections, key=lambda d: d.score[0])
            bb = det.location_data.relative_bounding_box
            return (int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H))

    return None

def feathered_oval_mask(h, w, rx=0.55, ry=0.70, feather=12):
    """
    Create a soft oval mask (0..1) centered in (h,w).
    rx/ry: relative radii (fraction of width/height) of the oval.
    feather: Gaussian sigma in pixels for edge softness.
    """
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    nx = (X - cx) / (rx * w / 2.0 + 1e-6)
    ny = (Y - cy) / (ry * h / 2.0 + 1e-6)
    mask = (nx * nx + ny * ny) <= 1.0
    mask = mask.astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        mask = np.clip(mask, 0.0, 1.0)
    return mask

def detect_face_box(gray, min_size=(80,80)):
    """
    OpenCV Haar cascade: returns largest face (x,y,w,h) or None.
    """
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=min_size)
    if len(faces) == 0:
        return None
    # pick largest by area
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return (x, y, w, h)

def to_square_biased(x, y, w, h, pad=1.25, y_bias=1):
    """
    Expand the rect to a square with padding, and shift vertically by y_bias*s.
    Negative y_bias moves up (toward forehead), helping exclude shoulders.
    """
    cx, cy = x + w / 2.0, y + h / 2.0
    s = max(w, h) * pad
    cy = cy + y_bias * s
    x2 = int(round(cx - s / 2.0))
    y2 = int(round(cy - s / 2.0))
    s = int(round(s))
    return x2, y2, s

def clamp_square(x, y, s, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    s = max(1, min(s, min(W - x, H - y)))
    return x, y, s

def match_luma_stats(base_y, patch_y, mask):
    """
    Scale/shift patch luma to match base luma under the mask region.
    """
    m = mask > 1e-3
    if np.count_nonzero(m) < 100:
        return patch_y
    base_mean = float(base_y[m].mean())
    base_std  = float(base_y[m].std() + 1e-6)
    patch_mean = float(patch_y[m].mean())
    patch_std  = float(patch_y[m].std() + 1e-6)
    # scale + shift, clamp to [0,1]
    gamma = base_std / patch_std
    beta  = base_mean - gamma * patch_mean
    y_adj = np.clip(gamma * patch_y + beta, 0.0, 1.0)
    return y_adj

def composite_face(base_bgr, patch_bgr, roi, rx=0.55, ry=0.70, feather=12, brightness_match=True):
    """
    Blend the patch into base at roi=(x,y,s). Y-only replacement with soft oval mask.
    """
    x, y, s = roi
    H, W, _ = base_bgr.shape
    if s <= 1:  # nothing to do
        return base_bgr

    # resize patch to ROI size
    patch = cv2.resize(patch_bgr, (s, s), interpolation=cv2.INTER_CUBIC)

    # Convert base region & patch to YUV
    base_yuv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV)
    patch_yuv = cv2.cvtColor(patch,     cv2.COLOR_BGR2YUV)

    # Extract Y planes and normalize to 0..1
    y_base = base_yuv[y:y+s, x:x+s, 0].astype(np.float32) / 255.0
    y_patch = patch_yuv[:, :, 0].astype(np.float32) / 255.0

    # Oval mask
    alpha = feathered_oval_mask(s, s, rx=rx, ry=ry, feather=feather)

    # Optional brightness/contrast match under mask
    if brightness_match:
        y_patch = match_luma_stats(y_base, y_patch, alpha)

    # Blend Y only; keep U,V from base
    y_mix = (1.0 - alpha) * y_base + alpha * y_patch
    base_yuv[y:y+s, x:x+s, 0] = np.clip(y_mix * 255.0, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(base_yuv, cv2.COLOR_YUV2BGR)
    return out


# --- Landmark-based alignment helpers --------------------------------------

def mp_keypoints_bgr(img_bgr, min_conf=0.3):
    """Return (left_eye, right_eye, nose_tip) keypoints in pixel coords using MediaPipe.
    Returns np.ndarray shape (3,2) or None if not available."""
    if not HAS_MP:
        return None
    H, W = img_bgr.shape[:2]
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_conf) as mpfd:
        res = mpfd.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    det = max(res.detections, key=lambda d: d.score[0])
    kps = det.location_data.relative_keypoints  # [right_eye, left_eye, nose_tip, mouth, right_ear, left_ear]
    left  = (kps[1].x * W, kps[1].y * H)
    right = (kps[0].x * W, kps[0].y * H)
    nose  = (kps[2].x * W, kps[2].y * H)
    return np.array([left, right, nose], dtype=np.float32)


def estimate_similarity(src_pts, dst_pts):
    """Estimate a 2x3 similarity transform mapping src_pts -> dst_pts using LMEDS."""
    if src_pts is None or dst_pts is None or len(src_pts) != 3 or len(dst_pts) != 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2), method=cv2.LMEDS)
    return M


def warp_patch_and_mask_to_frame(patch_bgr, frame_shape, M, mask_rx=0.55, mask_ry=0.70, feather=12):
    """Apply affine M to the patch and to an oval mask; return (warped_patch, warped_mask in 0..1)."""
    H, W = frame_shape[:2]
    warped_patch = cv2.warpAffine(patch_bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    pmask = feathered_oval_mask(patch_bgr.shape[0], patch_bgr.shape[1], rx=mask_rx, ry=mask_ry, feather=feather)
    pmask = (pmask * 255.0).astype(np.uint8)
    warped_mask = cv2.warpAffine(pmask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = warped_mask.astype(np.float32) / 255.0
    warped_mask = np.clip(warped_mask, 0.0, 1.0)
    return warped_patch, warped_mask


def composite_affine_y(base_bgr, warped_bgr, alpha_mask, brightness_match=True):
    """Y-only composite in full-frame space using a pre-warped patch and mask."""
    base_yuv  = cv2.cvtColor(base_bgr,  cv2.COLOR_BGR2YUV)
    patch_yuv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2YUV)
    y_base  = base_yuv[:, :, 0].astype(np.float32) / 255.0
    y_patch = patch_yuv[:, :, 0].astype(np.float32) / 255.0
    a = alpha_mask.astype(np.float32)
    if brightness_match and a.mean() > 1e-3:
        m = a > 1e-3
        base_mean = float(y_base[m].mean()); base_std = float(y_base[m].std() + 1e-6)
        patch_mean = float(y_patch[m].mean()); patch_std = float(y_patch[m].std() + 1e-6)
        gamma = base_std / patch_std; beta = base_mean - gamma * patch_mean
        y_patch = np.clip(gamma * y_patch + beta, 0.0, 1.0)
    y_mix = (1.0 - a) * y_base + a * y_patch
    base_yuv[:, :, 0] = (np.clip(y_mix, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(base_yuv, cv2.COLOR_YUV2BGR)

def main():
    ap = argparse.ArgumentParser(description="Overlay a 384x384 HD face patch onto a blurry image (face ROI auto-detected).")
    ap.add_argument("--base", required=True, help="Path to blurry base image (e.g., frame.jpg)")
    ap.add_argument("--patch", required=True, help="Path to HD face patch image (ideally 384x384)")
    ap.add_argument("--out", default="composited.jpg", help="Output path")
    ap.add_argument("--pad", type=float, default=1.25, help="Padding around detected face (1.2–1.4)")
    ap.add_argument("--y_bias", type=float, default=-0.10, help="Negative = shift box up (exclude shoulders)")
    ap.add_argument("--min_face", type=int, default=120, help="Minimum detected face size (pixels)")
    ap.add_argument("--rx", type=float, default=0.55, help="Oval mask horizontal radius fraction")
    ap.add_argument("--ry", type=float, default=0.70, help="Oval mask vertical radius fraction")
    ap.add_argument("--feather", type=int, default=12, help="Gaussian blur sigma for mask edge")
    ap.add_argument("--no_brightness_match", action="store_true", help="Disable Y-channel brightness match")
    # --- Added arguments for robust detection ---
    ap.add_argument("--aggressive", action="store_true", help="Use aggressive Haar settings for blurry faces")
    ap.add_argument("--mediapipe", action="store_true", help="Try MediaPipe fallback if installed")
    ap.add_argument("--boost", type=float, default=1.5, help="Pre-upsampling factor before detection (1.0 disables)")
    ap.add_argument("--clahe", action="store_true", help="Use CLAHE instead of global hist eq for detection")
    ap.add_argument("--select", action="store_true", help="Interactive ROI select if detection fails")
    ap.add_argument("--roi_xys", type=int, nargs=3, metavar=("X","Y","S"), help="Manual ROI square if known")
    ap.add_argument("--align", dest="align", action="store_true", help="Use landmark-based alignment (MediaPipe) before compositing")
    ap.add_argument("--no_align", dest="align", action="store_false", help="Disable landmark alignment and use ROI resize")
    ap.set_defaults(align=True)
    ap.add_argument("--debug", action="store_true", help="Write a _debug.jpg with landmarks overlay")
    # --- Offset controls ---
    ap.add_argument("--offset_y_pct", type=float, default=0.0, help="Post-align vertical offset as fraction of size (ROI size for ROI path; min(frame W,H) for aligned path). Positive moves down.")
    ap.add_argument("--offset_x_pct", type=float, default=0.0, help="Post-align horizontal offset as fraction of size. Positive moves right.")
    ap.add_argument("--offset_y_px", type=int, default=0, help="Post-align vertical offset in pixels (added after pct). Positive moves down.")
    ap.add_argument("--offset_x_px", type=int, default=0, help="Post-align horizontal offset in pixels (added after pct). Positive moves right.")
    ap.add_argument("--probe", action="store_true", help="Detect-only mode: print bbox and centroid; save annotated image to --out")
    ap.add_argument("--probe_draw", action="store_true", help="Draw bbox/centroid/landmarks in detect-only mode")
    args = ap.parse_args()

    base = cv2.imread(args["base"] if isinstance(args, dict) else args.base, cv2.IMREAD_COLOR)
    patch = cv2.imread(args["patch"] if isinstance(args, dict) else args.patch, cv2.IMREAD_COLOR)
    if base is None:
        raise SystemExit("Could not read base image.")
    if patch is None:
        raise SystemExit("Could not read patch image.")

    H, W, _ = base.shape

    # Face detection on grayscale base
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    # Detection with fallbacks
    if args.roi_xys:
        x, y, s = args.roi_xys
        x, y, s = clamp_square(x, y, s, W, H)
        det = (x, y, s, s)
    else:
        det = detect_face_any(
            gray,
            min_face=args.min_face,
            aggressive=args.aggressive,
            try_mediapipe=args.mediapipe,
            boost=args.boost,
            use_clahe=args.clahe,
        )

    if det is None and args.select:
        box = cv2.selectROI("Select face (square-ish)", base, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if box != (0,0,0,0):
            det = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    if det is None:
        # final fallback: center square
        s = min(W, H) // 2
        x = (W - s) // 2; y = (H - s) // 2
        print("WARN: no face detected; using centered ROI")
        det = (x, y, s, s)

    x, y, w, h = det
    x, y, s = to_square_biased(x, y, w, h, pad=args.pad, y_bias=args.y_bias)
    x, y, s = clamp_square(x, y, s, W, H)

    # ------------------------------------------------------------
    # Probe-only mode: print bbox + centroid and optionally draw
    # ------------------------------------------------------------
    if args.probe:
        cx = int(x + w/2)
        cy = int(y + h/2)
        print(f"DETECT bbox: x={x}, y={y}, w={w}, h={h}")
        print(f"DETECT centroid: cx={cx}, cy={cy}")
        print(f"ROI square: x={x}, y={y}, s={s}")

        if args.probe_draw:
            vis = base.copy()
            # raw detection rectangle
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # ROI square (biased/expanded)
            cv2.rectangle(vis, (x, y), (x+s, y+s), (255, 165, 0), 2)
            # centroid
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            # optional landmarks if available
            if HAS_MP:
                kps = mp_keypoints_bgr(base, min_conf=0.3)
                if kps is not None:
                    for (kx, ky) in kps:
                        cv2.circle(vis, (int(kx), int(ky)), 3, (255, 0, 255), -1)
            out_path = args.out if args.out else "probe_out.jpg"
            cv2.imwrite(out_path, vis)
            print(f"Annotated image saved → {out_path}")
        return

    # Try landmark-based alignment first
    if args.align and HAS_MP:
        base_kp  = mp_keypoints_bgr(base, min_conf=0.3)
        patch_kp = mp_keypoints_bgr(patch, min_conf=0.3)
        M = estimate_similarity(patch_kp, base_kp)
        if M is not None:
            # Apply post-align offsets
            M = M.copy()
            base_min = min(base.shape[0], base.shape[1])
            dx = int(args.offset_x_px + args.offset_x_pct * base_min)
            dy = int(args.offset_y_px + args.offset_y_pct * base_min)
            M[0, 2] += dx
            M[1, 2] += dy
            warped_patch, warped_mask = warp_patch_and_mask_to_frame(
                patch, base.shape, M, mask_rx=args.rx, mask_ry=args.ry, feather=args.feather)
            out = composite_affine_y(base, warped_patch, warped_mask, brightness_match=(not args.no_brightness_match))
            if args.debug and base_kp is not None:
                dbg = out.copy()
                for p in base_kp:
                    cv2.circle(dbg, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
                cv2.imwrite(os.path.splitext(args.out)[0] + "_debug.jpg", dbg)
            cv2.imwrite(args.out, out)
            print(f"Saved → {args.out} (aligned)\nROI: x={x}, y={y}, s={s} | offsets(px): x={args.offset_x_px}, y={args.offset_y_px} | offsets(pct): x={args.offset_x_pct}, y={args.offset_y_pct}")
            return
        else:
            print("WARN: landmark alignment failed; using ROI-based composite")

    # Apply offsets in ROI space
    dx_roi = int(args.offset_x_px + args.offset_x_pct * s)
    dy_roi = int(args.offset_y_px + args.offset_y_pct * s)
    x = max(0, min(x + dx_roi, W - s))
    y = max(0, min(y + dy_roi, H - s))
    # Fallback: ROI-based composite
    out = composite_face(
        base_bgr=base,
        patch_bgr=patch,
        roi=(x, y, s),
        rx=args.rx, ry=args.ry,
        feather=args.feather,
        brightness_match=(not args.no_brightness_match)
    )
    cv2.imwrite(args.out, out)
    print(f"Saved → {args.out}\nROI: x={x}, y={y}, s={s} | offsets(px): x={args.offset_x_px}, y={args.offset_y_px} | offsets(pct): x={args.offset_x_pct}, y={args.offset_y_pct}")

if __name__ == "__main__":
    main()