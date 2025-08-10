#!/usr/bin/env python3
import argparse, os, sys
import cv2
import numpy as np

import insightface
from insightface.app import FaceAnalysis

def pick_largest(faces):
    if not faces: return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def main():
    ap = argparse.ArgumentParser(description="Swap/overlay a source face onto a target image using InsightFace inswapper.")
    ap.add_argument("--src", required=True, help="Path to source face image (HD/reference).")
    ap.add_argument("--dst", required=True, help="Path to target image (blurry frame).")
    ap.add_argument("--out", default="out.jpg", help="Output path.")
    ap.add_argument("--model", default="inswapper_128.onnx", help="Swapper model name or path.")
    ap.add_argument("--det_size", type=int, default=640, help="Detection size (width=height).")
    ap.add_argument("--provider", choices=["cpu","gpu"], default="cpu", help="ONNXRuntime execution provider.")
    ap.add_argument("--all_faces", action="store_true", help="Swap all faces in target instead of just the largest.")
    ap.add_argument("--dst_face_index", type=int, default=None, help="Swap only this index (0..N-1) from detected target faces.")
    ap.add_argument("--upsample", type=float, default=1.0, help="Optional pre-upsampling for very blurry target (e.g., 1.5).")
    args = ap.parse_args()

    # Load images
    src = cv2.imread(args.src);  dst = cv2.imread(args.dst)
    if src is None: sys.exit(f"Could not read --src: {args.src}")
    if dst is None: sys.exit(f"Could not read --dst: {args.dst}")

    # Optional pre-upsample for very blurry targets (helps detection)
    if args.upsample and args.upsample != 1.0:
        dst = cv2.resize(dst, None, fx=args.upsample, fy=args.upsample, interpolation=cv2.INTER_CUBIC)

    # Providers for ONNXRuntime
    providers = ["CPUExecutionProvider"] if args.provider == "cpu" else ["CUDAExecutionProvider","CPUExecutionProvider"]

    # Face detector/landmarks/embedding
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    # ctx_id: -1 CPU, >=0 GPU index. We let providers decide; set -1 for cross-platform safety.
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    # Swapper model (auto-downloads if name matches)
    swapper = insightface.model_zoo.get_model(args.model, providers=providers)

    # Detect source & target faces
    src_faces = app.get(src)
    if len(src_faces) == 0:
        sys.exit("No face found in --src. Use a clear reference photo (frontal if possible).")
    src_face = pick_largest(src_faces)

    dst_faces = app.get(dst)
    if len(dst_faces) == 0:
        sys.exit("No face found in --dst (even after optional upsample). Try --upsample 1.5 or a clearer frame.")

    # Choose which target faces to swap
    faces_to_swap = []
    if args.all_faces:
        faces_to_swap = list(range(len(dst_faces)))
    elif args.dst_face_index is not None:
        if 0 <= args.dst_face_index < len(dst_faces):
            faces_to_swap = [args.dst_face_index]
        else:
            sys.exit(f"--dst_face_index {args.dst_face_index} out of range (0..{len(dst_faces)-1}).")
    else:
        # default: largest
        largest_idx = int(np.argmax([(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in dst_faces]))
        faces_to_swap = [largest_idx]

    # Do the swap(s)
    out = dst.copy()
    for i in faces_to_swap:
        out = swapper.get(out, dst_faces[i], src_face, paste_back=True)

    # If we upsampled the target, downscale back to original size
    if args.upsample and args.upsample != 1.0:
        H, W = cv2.imread(args.dst).shape[:2]
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)

    cv2.imwrite(args.out, out)
    print(f"Saved → {args.out}")
    print(f"src faces: {len(src_faces)}, dst faces: {len(dst_faces)}, swapped indices: {faces_to_swap}")

if __name__ == "__main__":
    main()