#!/usr/bin/env python3
# Export SDXL VAE decoder to ONNX (FP32 + optional pure FP16)
# Usage:
#   python export_sdxl_vae_decoder.py --out sdxl_vae_decoder_fp32.onnx
#   python export_sdxl_vae_decoder.py --fp16 --out sdxl_vae_decoder_fp16.onnx
#   (You can run both once: it will write both files if --out not given.)

import argparse, os, sys, pathlib, torch
from diffusers import AutoencoderKL

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="stabilityai/sdxl-vae")
    ap.add_argument("--size", type=int, default=384, help="dummy image size (HxW) to set shapes; output is dynamic anyway")
    ap.add_argument("--out", type=str, default="", help="output filename; if empty, will write fp32 and/or fp16 defaults")
    ap.add_argument("--fp16", action="store_true", help="export a pure float16 ONNX (requires CUDA for tracing)")
    ap.add_argument("--no-simplify", action="store_true", help="skip onnx-simplifier")
    ap.add_argument("--opset", type=int, default=17)
    return ap.parse_args()

class VaeDecoder(torch.nn.Module):
    """Wrap SDXL decoder; expects z_scaled already multiplied by scaling_factor."""
    def __init__(self, vae, scale: float):
        super().__init__()
        self.dec = vae.decoder
        self.scale = float(scale)
    def forward(self, z_scaled):
        return self.dec(z_scaled / self.scale)  # returns [-1,1] in NCHW

def export_one(dec, dummy, out_path, opset=17, dynamic=True, simplify=True):
    out_path = str(out_path)
    print(f"[export] -> {out_path}")
    torch.onnx.export(
        dec, dummy, out_path,
        input_names=["z_scaled"], output_names=["y"],
        dynamic_axes={"z_scaled": {0:"B", 2:"H8", 3:"W8"},
                      "y":        {0:"B", 2:"H",  3:"W"}},
        do_constant_folding=True,
        opset_version=opset,
    )
    # Basic check
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print(f"[check] OK: {out_path}")

    if simplify:
        try:
            import onnxsim
            print("[simplify] running onnx-simplifier…")
            simp, ok = onnxsim.simplify(m, dynamic_input_shape=True)
            if ok:
                onnx.save(simp, out_path)
                print("[simplify] wrote simplified model.")
            else:
                print("[simplify] simplifier returned not-ok; keeping original graph.")
        except Exception as e:
            print(f"[simplify] skipped ({e.__class__.__name__}: {e})")

def main():
    args = parse_args()
    device = "cuda" if (args.fp16 and torch.cuda.is_available()) else "cpu"
    print(f"[load] {args.model_id} on device={device}")
    vae = AutoencoderKL.from_pretrained(args.model_id).to(device).eval()
    scale = float(vae.config.scaling_factor)
    print(f"[info] scaling_factor={scale}")

    # Wrap decoder with scaling baked-in
    dec = VaeDecoder(vae, scale).eval()

    # Dummy latent for shape: [1, 4, H/8, W/8]
    H8 = args.size // 8
    W8 = args.size // 8

    # --- FP32 export ---
    if not args.fp16 or (args.fp16 and not args.out):
        dec32 = dec.to("cpu").float().eval()
        dummy32 = torch.randn(1, 4, H8, W8, dtype=torch.float32)
        out32 = args.out or "sdxl_vae_decoder_fp32.onnx"
        export_one(dec32, dummy32, out32, opset=args.opset, simplify=not args.no_simplify)

    # --- Pure FP16 export (no post-hoc converter) ---
    if args.fp16:
        if device != "cuda":
            print("[warn] --fp16 requested but CUDA not available; skipping FP16 export.")
            return
        dec16 = dec.to("cuda").half().eval()
        dummy16 = torch.randn(1, 4, H8, W8, dtype=torch.float16, device="cuda")
        # warm-up to materialize kernels
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            _ = dec16(dummy16)
            torch.cuda.synchronize()
        out16 = args.out or "sdxl_vae_decoder_fp16.onnx"
        export_one(dec16, dummy16, out16, opset=args.opset, simplify=not args.no_simplify)

if __name__ == "__main__":
    main()