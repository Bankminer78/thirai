# export_coreml_decoder.py (ensure FP32 first)
import torch, coremltools as ct
import numpy as np
from torch import nn
from diffusers import AutoencoderKL

H = W = 384
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
scale = float(vae.config.scaling_factor)

class SDXLDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post = vae.post_quant_conv
        self.dec  = vae.decoder
        self.scale = scale
    def forward(self, z_scaled):                 # [1,4,H/8,W/8]
        z = z_scaled / self.scale
        z = self.post(z)
        y = self.dec(z)                          # [-1,1], NCHW
        return y

dec = SDXLDecoder(vae).eval()
ex  = torch.randn(1,4,H//8,W//8, dtype=torch.float32)
ts  = torch.jit.trace(dec, ex); ts = torch.jit.freeze(ts)

ml = ct.convert(
    ts,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    compute_units=ct.ComputeUnit.ALL,
    inputs=[ct.TensorType(name="z_scaled", shape=(1,4,H//8,W//8), dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32,  # <- start FP32 to establish parity
)
ml.save(f"sdxl_vae_decoder_{H}x{W}.mlpackage")
print("saved decoder")