# export_encoder_img_nchw.py — SDXL VAE encoder -> Core ML (ImageType input, NCHW)
import coremltools as ct
import torch, numpy as np
from torch import nn
from diffusers import AutoencoderKL

H = W = 384
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
s  = float(vae.config.scaling_factor)

class SDXLEncoderScaledNCHW(nn.Module):
    def __init__(self, vae, s):
        super().__init__()
        self.enc = vae.encoder
        self.qc  = vae.quant_conv
        self.s   = s
    def forward(self, x):                 # x: [1,3,H,W] in [-1,1]
        h  = self.qc(self.enc(x))         # [1,8,H/8,W/8]
        mu, _ = torch.chunk(h, 2, dim=1)  # SDXL uses mean of diag Gaussian
        return mu * self.s                # [1,4,H/8,W/8] (z_scaled)

m = SDXLEncoderScaledNCHW(vae, s).eval()
ex = torch.randn(1, 3, H, W, dtype=torch.float32)  # NCHW trace
ts = torch.jit.trace(m, ex); ts = torch.jit.freeze(ts)

ml = ct.convert(
    ts,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,   # or iOS17
    compute_units=ct.ComputeUnit.ALL,
    # Core ML expects (1,3,H,W) for ImageType inputs
    inputs=[ct.ImageType(
        name="x", shape=(1,3,H,W),
        color_layout=ct.colorlayout.RGB,
        # map 0..255 -> [-1,1] inside Core ML
        scale=1/127.5, bias=[-1.0, -1.0, -1.0]
    )],
    outputs=[ct.TensorType(name="z_scaled", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,        # encoder FP32 = fewer artifacts
)
ml.save("sdxl_vae_encoder_384x384.mlpackage")
print("saved encoder")