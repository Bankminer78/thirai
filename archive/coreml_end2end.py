# coreml_end2end_test.py (256, .mlpackage, double warmup to avoid cold start)
import time, json, os
import numpy as np
from PIL import Image
import torch, coremltools as ct
from diffusers import AutoencoderKL
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
H = W = 256
ENC_MLPATH = os.path.join(ROOT, f"sdxl_vae_encoder_{H}x{W}.mlpackage")
DEC_MLPATH = os.path.join(ROOT, f"sdxl_vae_decoder_{H}x{W}.mlpackage")
IMG_PATH   = os.path.join(ROOT, "crop.jpg")

# ---- helpers ----
def to01(pil):
    return np.asarray(pil).astype(np.float32)/255.0

def nchw01(img01):
    return np.transpose(img01, (2,0,1))[None,...].copy()

def psnr01(a,b):
    a = np.clip(a,0,1); b = np.clip(b,0,1)
    mse = float(((a-b)**2).mean())
    import math
    return 10*math.log10(1.0/max(mse,1e-12))

def ssim_y(x,y):
    # x,y: torch tensors [1,3,H,W] in [0,1]
    def toY(a): return 0.299*a[:,0:1]+0.587*a[:,1:2]+0.114*a[:,2:3]
    xY, yY = toY(x), toY(y)
    w = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=x.dtype, device=x.device); w/=w.sum()
    w = w.view(1,1,3,3)
    C1,C2 = 0.01**2, 0.03**2
    mux = F.conv2d(xY, w, padding=1); muy = F.conv2d(yY, w, padding=1)
    mux2, muy2, muxy = mux**2, muy**2, mux*muy
    sigx2 = F.conv2d(xY*xY,w,padding=1)-mux2
    sigy2 = F.conv2d(yY*yY,w,padding=1)-muy2
    sigxy = F.conv2d(xY*yY,w,padding=1)-muxy
    ssim = ((2*muxy+C1)*(2*sigxy+C2))/((mux2+muy2+C1)*(sigx2+sigy2+C2)+1e-8)
    return float(ssim.mean())

# ---- load models ----
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
scale = float(vae.config.scaling_factor)
enc_ml = ct.models.MLModel(ENC_MLPATH, compute_units=ct.ComputeUnit.ALL)
dec_ml = ct.models.MLModel(DEC_MLPATH, compute_units=ct.ComputeUnit.ALL)

# ---- image ----
pil = Image.open(IMG_PATH).convert("RGB").resize((W,H), Image.BICUBIC)
x01 = to01(pil); x_nchw01 = nchw01(x01); x = torch.from_numpy(x_nchw01)*2-1

# ---- PyTorch baseline ----
with torch.no_grad():
    z_pt = vae.encode(x).latent_dist.mean * scale      # [1,4,H/8,W/8]
    y_pt = vae.decode(z_pt/scale).sample               # [-1,1]
    y_pt01 = ((y_pt+1)/2).clamp(0,1)

# ---- CoreML encode (double warmup) ----
enc_in = x.numpy().astype(np.float32)
_ = enc_ml.predict({"x": enc_in})
_ = enc_ml.predict({"x": enc_in})
_t0=time.time(); enc_out = enc_ml.predict({"x": enc_in}); enc_ms=(time.time()-_t0)*1000
z_any = next(iter(enc_out.values()))
z_arr = np.array(z_any)
if z_arr.ndim != 4: raise RuntimeError(f"Unexpected latent rank from encoder: {z_arr.shape}")
if z_arr.shape[1] == 4:
    z_cm = z_arr.copy()                               # NCHW
else:
    z_cm = np.transpose(z_arr, (0,3,1,2)).copy()      # NHWC -> NCHW

# ---- CoreML decode (double warmup) ----
_ = dec_ml.predict({"z_scaled": z_cm})
_ = dec_ml.predict({"z_scaled": z_cm})
_t0=time.time(); dec_out = dec_ml.predict({"z_scaled": z_cm}); dec_ms=(time.time()-_t0)*1000
y_any = next(iter(dec_out.values()))
y_arr = np.array(y_any)
if y_arr.ndim != 4: raise RuntimeError(f"Unexpected decoder output rank: {y_arr.shape}")
if y_arr.shape[1] == 3:
    y_cm = y_arr.copy()                                # NCHW
else:
    y_cm = np.transpose(y_arr, (0,3,1,2)).copy()       # NHWC -> NCHW
# to [0,1]
y_cm01 = np.clip((y_cm+1)/2, 0, 1)

# ---- Metrics (CoreML vs PyTorch) ----
t_pt = y_pt01.detach().cpu().to(torch.float32)
t_cm = torch.from_numpy(y_cm01.astype(np.float32, copy=False))
mse = torch.mean((t_pt - t_cm)**2).item()
psnr = 10*np.log10(1.0/max(mse,1e-12))
ssim = ssim_y(t_pt, t_cm)
print(f"Encode (CoreML): {enc_ms:.2f} ms | Decode (CoreML): {dec_ms:.2f} ms")
print(f"PSNR vs PyTorch: {psnr:.2f} dB | SSIM: {ssim:.3f}")

# ---- Save latent for browser (fp32 for parity) ----
z_cm_np = z_cm.astype(np.float32, copy=False)
latent_bin = os.path.join(ROOT, f"latent_{H}_fp32.bin")
with open(latent_bin, "wb") as f:
    f.write(z_cm_np.tobytes())
with open(os.path.join(ROOT, f"latent_{H}_fp32.json"),"w") as jf:
    json.dump({"shape":[1,4,H//8,W//8],"dtype":"float32","scaled":True,"scaling":scale}, jf, indent=2)
print("Saved latent for browser:", latent_bin)

# ---- Optional: write side-by-side PNG for eyeballing ----
try:
    import imageio
    side = np.concatenate([
        (y_pt01.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8),
        (y_cm01[0].transpose(1,2,0)*255).astype(np.uint8)
    ], axis=1)
    imageio.imwrite(os.path.join(ROOT, f"e2e_side_{H}.png"), side)
    print(f"Saved side-by-side preview: e2e_side_{H}.png (left=PyTorch, right=CoreML)")
except Exception as e:
    print("[warn] failed to write side-by-side preview:", e)