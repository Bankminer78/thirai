# export_coreml_encoder.py (re-export)
import argparse
import torch, coremltools as ct
import numpy as np
from torch import nn
from diffusers import AutoencoderKL

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=384, help="Input resolution H=W")
args = ap.parse_args()

H = W = int(args.size)
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
scale = float(vae.config.scaling_factor)

class SDXLEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.enc = vae.encoder
        self.q   = vae.quant_conv   # produces [B, 2*C, H/8, W/8]
        self.scale = scale
    def forward(self, x):           # x in [-1,1], NCHW
        h = self.enc(x)
        m = self.q(h)
        mean, logvar = torch.chunk(m, 2, dim=1)
        z_scaled = mean * self.scale
        return z_scaled

enc = SDXLEncoder(vae).eval()
ex  = torch.randn(1,3,H,W, dtype=torch.float32)
ts  = torch.jit.trace(enc, ex); ts = torch.jit.freeze(ts)

ml = ct.convert(
    ts,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    compute_units=ct.ComputeUnit.ALL,
    inputs=[ct.TensorType(name="x", shape=(1,3,H,W), dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,  # encoder FP16 is fine
)
ml.save(f"sdxl_vae_encoder_{H}x{W}.mlpackage")
print("saved encoder", f"sdxl_vae_encoder_{H}x{W}.mlpackage")