#!/usr/bin/env python3
# ae_modal_app.py
#
# One-file Modal app:
# - Train a deterministic Autoencoder (AE) for 384×384 face crops (Y-only, f=32)
# - Serve an inference endpoint: POST a 384×384 crop, get reconstruction (PNG)
#
# Quickstart (terminal):
#   python3 -m venv venv && source venv/bin/activate
#   pip install modal
#   modal setup
#
#   # 1) Stage your dataset (must contain hr_train/ and hr_val/ with 384x384 PNG/JPG)
#   modal run ae_modal_app.py::stage_data --local_dir myface_ds --remote_name myface_ds
#
#   # 2) Train on GPU (A10G/L4). Adjust epochs/batch if needed.
#   modal run ae_modal_app.py::train_job --data_name myface_ds --run_name ae384_run --epochs 10 --bs 64
#
#   # 3) Deploy web endpoint
#   modal deploy ae_modal_app.py
#   curl -X POST -F "file=@face_384.png" "https://<your-app>.modal.run/reconstruct?roi=384" --output recon.png
#
# Dataset expectation:
#   /artifacts/datasets/<data_name>/
#       hr_train/*.png|jpg   # 384x384 face crops
#       hr_val/*.png|jpg     # 384x384 face crops

import io, os, math, glob, random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import modal
from fastapi import UploadFile, File, Response

# =========================
#   Model: tiny AE (f=32)
# =========================

class DW(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=True)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x): return F.silu(self.bn(self.pw(self.dw(x))))

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = DW(cin, cout)
        self.c2 = DW(cout, cout)
        self.ds = nn.Conv2d(cout, cout, 3, 2, 1)  # stride-2
    def forward(self, x):
        x = self.c1(x); x = self.c2(x); return self.ds(x)

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = DW(cin, cout)
        self.c2 = DW(cout, cout)
    def forward(self, x):
        x = self.up(x); x = self.c1(x); return self.c2(x)

class TinyAE(nn.Module):
    """
    Deterministic autoencoder, Y-only, f=32 (5 stride-2 downs).
    384x384 -> latent 12x12. Keep channels modest for speed.
    """
    def __init__(self, ch=64, in_ch=1):
        super().__init__()
        self.e1 = Down(in_ch, ch)        # /2
        self.e2 = Down(ch, ch*2)         # /4
        self.e3 = Down(ch*2, ch*3)       # /8
        self.e4 = Down(ch*3, ch*4)       # /16
        self.e5 = Down(ch*4, ch*4)       # /32  (bottleneck)
        self.to_lat   = nn.Conv2d(ch*4, 64, 1)
        self.from_lat = nn.Conv2d(64, ch*4, 1)
        self.d5 = Up(ch*4, ch*4)         # x2
        self.d4 = Up(ch*4, ch*3)         # x2
        self.d3 = Up(ch*3, ch*2)         # x2
        self.d2 = Up(ch*2, ch)           # x2
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DW(ch, ch),
            nn.Conv2d(ch, in_ch, 3, 1, 1)
        )
    def forward(self, x):
        h = self.e1(x); h = self.e2(h); h = self.e3(h); h = self.e4(h); h = self.e5(h)
        z = self.to_lat(h)
        h = self.from_lat(z)
        h = self.d5(h); h = self.d4(h); h = self.d3(h); h = self.d2(h)
        y = self.d1(h)
        return y.clamp(0,1)

# =========================
#   Dataset (Y-only)
# =========================

def rgb_to_y(img: Image.Image, roi: int) -> torch.Tensor:
    """PIL RGB -> luma Y resized to roi x roi, returns [1,1,H,W] in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    y = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.float32)
    y_img = Image.fromarray((y*255).astype(np.uint8)).resize((roi, roi), Image.BICUBIC)
    y_np = np.asarray(y_img, dtype=np.float32) / 255.0
    return torch.from_numpy(y_np)[None, None, :, :]

class FaceCrops(Dataset):
    def __init__(self, root: str, roi: int = 384, augment: bool = True):
        exts = ("*.png","*.jpg","*.jpeg","*.webp")
        files = []
        for pat in exts:
            files.extend(sorted(Path(root).glob(pat)))
        if not files:
            raise FileNotFoundError(f"No images found in {root}")
        self.files = files
        self.roi = roi
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        # mild, cheap augmentations to mimic compression/blur
        if self.augment:
            # random tiny blur
            if random.random() < 0.25:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
            # random JPEG re-encode (compression artifact)
            if random.random() < 0.25:
                q = random.randint(15, 35)
                buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
                img = Image.open(buf).convert("RGB")
        y = rgb_to_y(img, self.roi)  # [1,1,H,W]
        return y

# =========================
#   Training utilities
# =========================

def train_loop(model: TinyAE, dl_train: DataLoader, dl_val: DataLoader,
               out_dir: str, epochs: int = 10, lr: float = 1e-3, device: torch.device = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float("inf")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"train e{ep}/{epochs}")
        running = 0.0
        for x in pbar:
            x = x.to(device, non_blocking=True)
            y = model(x)
            loss = F.l1_loss(y, x)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / max(1, len(dl_train))

        # val
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xv in dl_val:
                xv = xv.to(device, non_blocking=True)
                yv = model(xv)
                vloss += F.l1_loss(yv, xv).item()
        val_loss = vloss / max(1, len(dl_val))

        # save
        torch.save(model.state_dict(), str(Path(out_dir)/f"ae_last.pt"))
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), str(Path(out_dir)/f"ae_best.pt"))

        print(f"[epoch {ep}] train {train_loss:.4f} | val {val_loss:.4f} | best {best:.4f}")

# =========================
#   Modal scaffolding
# =========================

app = modal.App("ae384-app")

# CUDA 12.1 wheels (works on L4/A10G/A100 images). If this fails, drop the extra index to get CPU wheels.
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("tqdm", "pillow", "numpy", "fastapi")
)

# One persistent volume to store datasets and outputs
vol = modal.Volume.from_name("ae-artifacts", create_if_missing=True)

# ---------- Training job ----------

@app.function(image=image, gpu=modal.gpu.L4(count=1), volumes={"/artifacts": vol}, timeout=60*60*3)
def train_job(
    data_name: str,
    run_name: str = "ae384_run",
    roi: int = 384,
    epochs: int = 10,
    bs: int = 64,
    lr: float = 1e-3,
    workers: int = 4,
):
    """
    Train on /artifacts/datasets/<data_name>/{hr_train,hr_val}, write to /artifacts/runs/<run_name>.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root = Path("/artifacts/datasets")/data_name
    train_dir = root/"hr_train"
    val_dir   = root/"hr_val"
    out_dir   = Path("/artifacts/runs")/run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = FaceCrops(str(train_dir), roi=roi, augment=True)
    ds_va = FaceCrops(str(val_dir),   roi=roi, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)

    model = TinyAE(ch=64, in_ch=1)
    train_loop(model, dl_tr, dl_va, str(out_dir), epochs=epochs, lr=lr, device=device)
    print(f"Done. Checkpoints at {out_dir}/ae_best.pt and ae_last.pt")

# ---------- Inference web endpoint ----------

MODEL: Optional[TinyAE] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model(ckpt_path: str) -> TinyAE:
    global MODEL
    if MODEL is not None:
        return MODEL
    m = TinyAE(ch=64, in_ch=1).to(DEVICE).eval()
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(state, strict=True)
    MODEL = m
    return MODEL

def _y_to_png(y: torch.Tensor) -> bytes:
    y_np = (y.squeeze().clip(0,1).cpu().numpy()*255.0).astype(np.uint8)
    out = Image.fromarray(y_np)
    buf = io.BytesIO(); out.save(buf, format="PNG"); return buf.getvalue()

@app.function(image=image, gpu=modal.gpu.L4(count=1), volumes={"/artifacts": vol}, timeout=60)
@modal.web_endpoint(method="POST")
async def reconstruct(
    file: UploadFile = File(...),
    roi: int = 384,
    data_name: str = "myface_ds",
    run_name: str = "ae384_run",
) -> Response:
    """
    POST an image file (face crop), returns AE reconstruction PNG (Y-only).
    Example:
      curl -X POST -F "file=@face_384.png" "https://<your-app>.modal.run/reconstruct?roi=384" --output recon.png
    """
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Where weights live:
    ckpt = Path("/artifacts/runs")/run_name/"ae_best.pt"
    model = _load_model(str(ckpt))
    with torch.no_grad():
        x = rgb_to_y(img, roi=roi).to(DEVICE)
        y = model(x)
    return Response(content=_y_to_png(y), media_type="image/png")

# ---------- Local helpers for data staging ----------

@app.local_entrypoint()
def stage_data(local_dir: str, remote_name: str):
    """
    Upload a local dataset folder into the Modal volume.

    local_dir layout (must exist):
        myface_ds/
          hr_train/*.png|jpg  (384x384)
          hr_val/*.png|jpg    (384x384)

    Example:
      modal run ae_modal_app.py::stage_data --local_dir myface_ds --remote_name myface_ds
    """
    src = Path(local_dir)
    if not src.exists():
        raise SystemExit(f"Local dir not found: {local_dir}")
    dst = f"/artifacts/datasets/{remote_name}"
    print(f"Uploading {src} -> {dst} (this may take a few minutes)...")
    vol.put_dir(str(src), dst)
    print("Done.")

@app.local_entrypoint()
def download_run(run_name: str = "ae384_run", out_dir: str = "downloaded_runs"):
    """
    Download training outputs (checkpoints) to your machine.
    Example:
      modal run ae_modal_app.py::download_run --run_name ae384_run --out_dir runs_out
    """
    dst = Path(out_dir); dst.mkdir(parents=True, exist_ok=True)
    vol.download_to(str(dst))
    print(f"Downloaded volume -> {dst}")