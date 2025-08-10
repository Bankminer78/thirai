# exportd_imag.py  (decoder -> ImageType output; no output shape)
import coremltools as ct
import torch
import numpy as np
from torch import nn
from diffusers import AutoencoderKL

H = W = 384  # pick the side you exported the encoder with
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
scale = float(vae.config.scaling_factor)

class SDXLDecoder(nn.Module):
    def __init__(self, vae, scale):
        super().__init__()
        self.post = vae.post_quant_conv
        self.dec  = vae.decoder
        self.s    = scale
    def forward(self, z_scaled):  # [1,4,H/8,W/8] float
        z = z_scaled / self.s
        z = self.post(z)
        y = self.dec(z)           # [-1,1], NCHW
        return y                  # Core ML will map to uint8 via outputs=ImageType

dec = SDXLDecoder(vae, scale).eval()
ex  = torch.randn(1,4,H//8,W//8, dtype=torch.float32)
ts  = torch.jit.trace(dec, ex)
ts  = torch.jit.freeze(ts)

ml = ct.convert(
    ts,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    compute_units=ct.ComputeUnit.ALL,
    inputs=[ct.TensorType(name="z_scaled", shape=(1,4,H//8,W//8), dtype=np.float32)],
    # NOTE: no `shape=` here for outputs — let Core ML infer it
    outputs=[ct.ImageType(
        name="y_img",
        channel_first=True,
        color_layout=ct.colorlayout.RGB,
        # map [-1,1] -> [0,255] in-graph: out = y*127.5 + 127.5
        scale=127.5, bias=[127.5,127.5,127.5],
    )],
    compute_precision=ct.precision.FLOAT16,  # FP16 decoder for speed; keep encoder FP32
)

ml.save(f"sdxl_vae_decoder_{H}x{W}_img.mlpackage")
print("saved decoder img-out")