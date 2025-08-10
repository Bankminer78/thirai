import time, os, json, argparse
import numpy as np
import coremltools as ct
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Run CoreML SDXL encoder and save latent .bin")
    p.add_argument("--img", type=str, default="crop.jpg", help="input RGB image path")
    p.add_argument("--size", type=int, default=384, help="square resize (matches model)")
    p.add_argument("--model", type=str, default=None, help="path to .mlpackage (defaults to sdxl_vae_encoder_{size}x{size}.mlpackage)")
    p.add_argument("--out", type=str, default=None, help="output .bin path (defaults to latent_{size}_fp{16|32}.bin)")
    p.add_argument("--dtype", type=str, choices=["fp32","fp16"], default="fp32", help="latent dtype to save")
    p.add_argument("--save_json", action="store_true", help="also save latent metadata JSON next to bin")
    return p.parse_args()


def load_image(path: str, size: int):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x01 = np.asarray(img).astype(np.float32) / 255.0   # [H,W,3] in [0,1]
    x = x01 * 2.0 - 1.0                                # [-1,1]
    # Model expects NCHW: [1,3,H,W]
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x, dtype=np.float32), img


def main():
    args = parse_args()
    H = W = int(args.size)

    model_path = args.model or f"sdxl_vae_encoder_{H}x{W}.mlpackage"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ml = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)

    x, pil = load_image(args.img, H)

    # Warmup
    for _ in range(2):
        ml.predict({"x": x})

    t0 = time.time()
    out = ml.predict({"x": x})
    ms = (time.time() - t0) * 1000.0

    # Pull first output and handle either NHWC or NCHW layouts
    z_any = next(iter(out.values()))
    z_arr = np.array(z_any)  # ensure ndarray
    if z_arr.ndim != 4:
        raise RuntimeError(f"Unexpected latent rank from model: {z_arr.shape}")

    # If channel==4 at dim=1 => NCHW; if channel==4 at last dim => NHWC
    if z_arr.shape[1] == 4:
        # NCHW already
        z_nchw = z_arr.copy()
        H8, W8 = z_nchw.shape[2], z_nchw.shape[3]
    elif z_arr.shape[-1] == 4:
        # NHWC -> NCHW
        z_nchw = np.transpose(z_arr, (0, 3, 1, 2)).copy()
        H8, W8 = z_nchw.shape[2], z_nchw.shape[3]
    else:
        raise RuntimeError(f"Unexpected latent shape from model (no 4-channel axis): {z_arr.shape}")

    # Choose dtype and save
    if args.dtype == "fp16":
        z_save = z_nchw.astype(np.float16, copy=False)
        suffix = "fp16"
    else:
        z_save = z_nchw.astype(np.float32, copy=False)
        suffix = "fp32"

    out_path = args.out or f"latent_{H}x{W}_{suffix}.bin"
    with open(out_path, "wb") as f:
        f.write(z_save.tobytes(order="C"))

    print(f"Model: {os.path.basename(model_path)}")
    print(f"Image: {args.img} -> resized {H}x{W}")
    print(f"Encode time: {ms:.2f} ms")
    print(f"Latent (NCHW): {z_nchw.shape}, dtype={z_save.dtype}")
    print(f"Saved: {out_path}  | bytes={os.path.getsize(out_path)}")
    print(f"Decoder expects: [1,4,{H//8},{W//8}] and dtype {suffix}")

    if args.save_json:
        meta = {
            "shape": [int(s) for s in z_nchw.shape],
            "dtype": "float16" if args.dtype == "fp16" else "float32",
            "H": H, "W": W,
            "H8": int(H/8), "W8": int(W/8),
            "note": "Values are z_scaled (scale already applied). Layout is NCHW, planar channels."
        }
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w") as jf:
            json.dump(meta, jf, indent=2)
        print(f"Saved metadata: {json_path}")


if __name__ == "__main__":
    main()