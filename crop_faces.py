import cv2, numpy as np, pandas as pd, argparse, os, math
import sys
try:
    import mediapipe as mp
    HAS_MP = True
except Exception:
    HAS_MP = False

def ema(prev, curr, a=0.7): return int(a*prev + (1-a)*curr) if prev is not None else int(curr)

def to_square(x,y,w,h, pad=1.25, y_bias=0.0):
    # y_bias < 0 shifts box upward by % of size
    cx, cy = x + w/2, y + h/2
    s = max(w, h) * pad
    cy += y_bias * s
    return int(cx - s/2), int(cy - s/2), int(s)

def clamp_rect(x,y,s,W,H):
    x, y = max(0,x), max(0,y)
    s = min(s, W - x, H - y)
    return x,y,s

def blur_score(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Robust detector that works on blurrier frames and can fall back to MediaPipe
def detect_face_any(gray, min_face=80, aggressive=False, try_mediapipe=False, boost=1.5, use_clahe=False):
    H, W = gray.shape[:2]
    # Pre-boost for blur: upsample and equalize
    g = cv2.resize(gray, None, fx=boost, fy=boost, interpolation=cv2.INTER_CUBIC) if boost and boost != 1.0 else gray.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    else:
        g = cv2.equalizeHist(g)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Normal pass
    params = dict(scaleFactor=1.2, minNeighbors=5, minSize=(min_face, min_face))
    faces = cascade.detectMultiScale(g, **params)
    # Aggressive pass if needed
    if len(faces) == 0 and aggressive:
        params = dict(scaleFactor=1.05, minNeighbors=2, minSize=(int(min_face*0.6), int(min_face*0.6)))
        faces = cascade.detectMultiScale(g, **params)

    if len(faces):
        x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
        if boost and boost != 1.0:
            x = int(x/boost); y = int(y/boost); w = int(w/boost); h = int(h/boost)
        return (x,y,w,h)

    # MediaPipe fallback (more robust to blur/low light)
    if try_mediapipe and HAS_MP:
        mpfd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        res = mpfd.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        if res.detections:
            det = max(res.detections, key=lambda d: d.score[0])
            bb = det.location_data.relative_bounding_box
            return (int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H))

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input .mov/.mp4 from QuickTime")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--roi", type=int, default=384, choices=[320,384,512,640])
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--pad", type=float, default=1.35)
    ap.add_argument("--min_blur", type=float, default=60.0)
    ap.add_argument("--y_bias", type=float, default=-0.10, help="Negative shifts box upward by % of size to avoid shoulders")
    ap.add_argument("--min_face", type=int, default=80, help="Minimum face size in pixels for detection")
    ap.add_argument("--aggressive", action="store_true", help="Use aggressive Haar settings for blurry faces")
    ap.add_argument("--mediapipe", action="store_true", help="Try MediaPipe fallback if installed")
    ap.add_argument("--boost", type=float, default=1.5, help="Pre-upsampling factor before detection (1.0 disables)")
    ap.add_argument("--clahe", action="store_true", help="Use CLAHE instead of global hist eq for detection")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/hr", exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise SystemExit("Could not open input video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps/args.fps))

    sm_x = sm_y = sm_s = None
    fidx = saved = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fidx % step != 0:
            fidx += 1; continue

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_face_any(
            gray_full,
            min_face=args.min_face,
            aggressive=args.aggressive,
            try_mediapipe=args.mediapipe,
            boost=args.boost,
            use_clahe=args.clahe,
        )
        if det is None:
            # If we have a previous smoothed ROI, reuse it; otherwise skip frame
            if sm_s is not None:
                x, y, s = sm_x, sm_y, sm_s
                # ensure ROI within bounds
                x, y, s = clamp_rect(x, y, s, W, H)
                roi = frame[y:y+s, x:x+s]
                if roi.size == 0:
                    fidx += 1; continue
            else:
                fidx += 1; continue
        else:
            x,y,w,h = det
            x,y,s = to_square(x,y,w,h, pad=args.pad, y_bias=args.y_bias)
            x,y,s = clamp_rect(x,y,s,W,H)
            sm_x = ema(sm_x, x); sm_y = ema(sm_y, y); sm_s = ema(sm_s, s)
            x,y,s = sm_x, sm_y, sm_s
            roi = frame[y:y+s, x:x+s]
            if roi.size == 0:
                fidx += 1; continue

        roi = cv2.resize(roi, (args.roi, args.roi), interpolation=cv2.INTER_CUBIC)
        bvar = blur_score(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        if bvar < args.min_blur:
            fidx += 1; continue

        out_path = f"{args.out}/hr/{saved:06d}.png"
        cv2.imwrite(out_path, roi, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        rows.append(dict(idx=saved, src_frame=fidx, time_ms=int(t_ms),
                         x=x, y=y, size=s, roi=args.roi, blur=float(bvar),
                         W=W, H=H, file=os.path.basename(out_path)))
        saved += 1
        fidx += 1

    cap.release()
    pd.DataFrame(rows).to_csv(f"{args.out}/meta.csv", index=False)
    print(f"Saved {saved} crops → {args.out}/hr and meta.csv (W={W}, H={H}, step={step}, boost={args.boost}, aggressive={args.aggressive}, mediapipe={args.mediapipe})")

if __name__ == "__main__":
    main()