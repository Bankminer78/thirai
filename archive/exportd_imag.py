# exportd_imag.py  — SDXL VAE decoder -> Core ML (ImageType output)
import coremltools as ct
import torch
import numpy as np
from torch import nn
from diffusers import AutoencoderKL

H = W = 384
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
scale = float(vae.config.scaling_factor)

class SDXLDecoderToU8(nn.Module):
    def __init__(self, vae, s):
        super().__init__()
        self.post = vae.post_quant_conv
        self.dec  = vae.decoder
        self.s    = s

    def forward(self, z_scaled):                      # [1,4,H/8,W/8], float
        z  = z_scaled / self.s
        z  = self.post(z)
        y  = self.dec(z)                              # [-1, 1], NCHW
        y  = torch.clamp(y, -1.0, 1.0)
        y  = (y + 1.0) * 127.5                        # [0, 255] float
        y  = torch.clamp(y, 0.0, 255.0)
        return y                                      # NCHW, RGB planes in [0..255]

dec = SDXLDecoderToU8(vae, scale).eval()
ex  = torch.randn(1, 4, H//8, W//8, dtype=torch.float32)
ts  = torch.jit.trace(dec, ex); ts = torch.jit.freeze(ts)

ml = ct.convert(
    ts,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,  # or iOS17
    compute_units=ct.ComputeUnit.ALL,
    inputs=[ct.TensorType(name="z_scaled", shape=(1,4,H//8,W//8), dtype=np.float32)],
    outputs=[ct.ImageType(
        name="y_img",
        color_layout=ct.colorlayout.RGB
    )],
    compute_precision=ct.precision.FLOAT16,         # FP16 decoder for speed
)

ml.save(f"sdxl_vae_decoder_{H}x{W}_img.mlpackage")
print("saved decoder img-out")