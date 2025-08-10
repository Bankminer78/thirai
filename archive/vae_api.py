from __future__ import annotations

import base64
import io
import os
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from diffusers import AutoencoderKL
from starlette.responses import Response, PlainTextResponse


# -------------------------
# Configuration
# -------------------------
MODEL_ID: str = os.environ.get("SDXL_VAE_MODEL", "stabilityai/sdxl-vae")
DEFAULT_SIZE: int = int(os.environ.get("VAE_SIZE", "384"))
DEFAULT_IMAGE_FORMAT: str = os.environ.get("VAE_IMAGE_FORMAT", "webp")  # webp|png
ALLOW_SAVE_TO_DISK: bool = os.environ.get("VAE_ALLOW_SAVE", "0") == "1"
SAVE_DIR: str = os.environ.get("VAE_SAVE_DIR", "./vae_runs")
DEFAULT_WEBP_QUALITY: int = int(os.environ.get("VAE_WEBP_QUALITY", "80"))  # speed vs size
DEFAULT_WEBP_METHOD: int = int(os.environ.get("VAE_WEBP_METHOD", "4"))      # 0..6, higher=slower

# Prebuilt resize/ToTensor for the default size (avoids re-allocs per request)
RESIZE_TF = transforms.Compose([
    transforms.Resize((DEFAULT_SIZE, DEFAULT_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])


# -------------------------
# Device selection (prefer macOS Metal/MPS)
# -------------------------
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Preferred model/latent dtype
_env_model_dtype = os.environ.get("VAE_MODEL_DTYPE", "auto").lower()  # auto|float16|float32
_env_latent_dtype = os.environ.get("VAE_LATENT_DTYPE", "auto").lower()  # auto|float16|float32

if _env_model_dtype == "float16":
    MODEL_DTYPE = torch.float16
elif _env_model_dtype == "float32":
    MODEL_DTYPE = torch.float32
else:
    # auto: FP16 only on CUDA by default; FP32 on MPS/CPU for correctness
    if DEVICE.type == "cuda":
        MODEL_DTYPE = torch.float16
    else:
        MODEL_DTYPE = torch.float32

if _env_latent_dtype == "float16":
    LATENT_DTYPE_STR = "float16"
elif _env_latent_dtype == "float32":
    LATENT_DTYPE_STR = "float32"
else:
    # auto aligns with model dtype for size/speed; we still prefer fp16 latents when model is fp16
    LATENT_DTYPE_STR = "float16" if MODEL_DTYPE == torch.float16 else "float32"


def _device_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# -------------------------
# Model load (singleton)
# -------------------------
vae: Optional[AutoencoderKL] = None
SCALING_FACTOR: Optional[float] = None
# Optional FP32 fallback model if primary model runs in FP16 or we want stable decode on MPS
vae_fp32: Optional[AutoencoderKL] = None


def load_model_once() -> None:
    global vae, vae_fp32, SCALING_FACTOR
    if vae is not None:
        return
    # Primary model
    vae = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE)
    vae = vae.to(DEVICE, dtype=MODEL_DTYPE).eval()
    # Prefer channels_last activations for conv-heavy nets
    vae = vae.to(memory_format=torch.channels_last)
    # Optional: slicing can reduce memory pressure
    try:
        vae.enable_slicing()
    except Exception:
        pass
    # CUDA-only performance knobs (no effect on MPS/CPU)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    SCALING_FACTOR = float(vae.config.scaling_factor)
    # Optional FP32 fallback for stability and MPS decode correctness
    try:
        _fallback = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        vae_fp32_local = _fallback.to(DEVICE, dtype=torch.float32).eval()
        try:
            vae_fp32_local.enable_slicing()
        except Exception:
            pass
        vae_fp32 = vae_fp32_local  # type: ignore[assignment]
    except Exception as e:
        print(f"[warn] Failed to prepare FP32 fallback model: {e}")


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="SDXL VAE Encode/Decode API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    load_model_once()
    os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Schemas
# -------------------------
class EncodeResponse(BaseModel):
    model_id: str
    device: str
    model_dtype: str
    latent_dtype: str
    size: int
    scaling: float
    shape: List[int]
    dtype: str
    enc_ms: float
    stats: dict
    latent_b64: str = Field(description="Base64 of contiguous Float16/Float32 tensor: [1,4,H/8,W/8]")


class DecodeRequest(BaseModel):
    latent_b64: str = Field(description="Base64 buffer (C-contiguous)")
    shape: List[int] = Field(description="Tensor shape, e.g. [1,4,H,W]")
    dtype: Optional[str] = Field(default=None, description="float16 or float32; defaults to server's latent dtype")
    scaling: Optional[float] = Field(default=None, description="Scaling factor used during encode; defaults to model's")
    output_format: str = Field(default=DEFAULT_IMAGE_FORMAT, description="webp or png")
    quality: int = Field(default=DEFAULT_WEBP_QUALITY, ge=10, le=100, description="WebP quality 10-100 (ignored for PNG)")
    webp_method: int = Field(default=DEFAULT_WEBP_METHOD, ge=0, le=6, description="WebP encoder method 0-6 (ignored for PNG)")


# -------------------------
# Helpers
# -------------------------

def _image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    # Reuse the prebuilt transform when using the default size
    if size == DEFAULT_SIZE:
        x01 = RESIZE_TF(image).unsqueeze(0)
    else:
        tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        x01 = tf(image).unsqueeze(0)

    # Move once, then set channels_last memory format for faster convs
    x01 = x01.to(DEVICE, dtype=torch.float32).contiguous(memory_format=torch.channels_last)
    x = (x01 * 2.0 - 1.0)  # [-1,1]
    x = x.to(DEVICE, dtype=MODEL_DTYPE).contiguous(memory_format=torch.channels_last)
    return x


def _tensor_to_image_bytes(y: torch.Tensor, fmt: str = "webp", quality: int = DEFAULT_WEBP_QUALITY, webp_method: int = DEFAULT_WEBP_METHOD):
    y01 = ((y + 1.0) / 2.0).float().clamp(0.0, 1.0)
    img_u8 = (
        y01[0]
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )  # HWC
    pil_img = Image.fromarray(img_u8)
    buf = io.BytesIO()
    fmt_lower = fmt.lower()
    if fmt_lower == "webp":
        pil_img.save(buf, format="WEBP", quality=int(quality), method=int(webp_method))
        media = "image/webp"
    elif fmt_lower == "png":
        pil_img.save(buf, format="PNG", optimize=True)
        media = "image/png"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported output_format '{fmt}'. Use 'webp' or 'png'.")
    buf.seek(0)
    return buf.getvalue(), media


def _encode_latent_to_b64(z: torch.Tensor, dtype_str: str) -> str:
    if dtype_str == "float16":
        z_cpu = z.detach().to("cpu", dtype=torch.float16).contiguous()
        arr = z_cpu.numpy()
    elif dtype_str == "float32":
        z_cpu = z.detach().to("cpu", dtype=torch.float32).contiguous()
        arr = z_cpu.numpy()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported latent dtype '{dtype_str}'")
    raw = arr.tobytes(order="C")
    return base64.b64encode(raw).decode("utf-8")


def _decode_b64_to_latent(latent_b64: str, shape: List[int], dtype_str: str) -> torch.Tensor:
    raw = base64.b64decode(latent_b64.encode("utf-8"))
    if dtype_str == "float16":
        np_dtype = np.float16
        torch_dtype = torch.float16
    elif dtype_str == "float32":
        np_dtype = np.float32
        torch_dtype = torch.float32
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported latent dtype '{dtype_str}'")
    arr = np.frombuffer(raw, dtype=np_dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise HTTPException(status_code=400, detail=f"latent size mismatch: buffer has {arr.size} elements, expected {expected} for shape {shape} and dtype {dtype_str}")
    arr = np.ascontiguousarray(arr.reshape(shape))
    z = torch.from_numpy(arr).to(DEVICE, dtype=torch_dtype)
    return z


# -------------------------
# Endpoints
# -------------------------
@app.get("/healthz")
def healthz() -> Response:
    load_model_once()
    return PlainTextResponse(
        f"ok | device={DEVICE} | model={MODEL_ID} | scaling={SCALING_FACTOR} | model_dtype={str(MODEL_DTYPE)} | latent_dtype={LATENT_DTYPE_STR}"
    )


@app.post("/encode", response_model=EncodeResponse)
async def encode_image(
    image: UploadFile = File(..., description="RGB image to encode"),
    size: int = Query(DEFAULT_SIZE, ge=64, le=2048, description="Square resize before encode"),
    save_to_disk: bool = Query(False, description="Save .pt package to disk (server-side)"),
) -> EncodeResponse:
    load_model_once()
    if vae is None or SCALING_FACTOR is None:
        raise HTTPException(status_code=500, detail="VAE not initialized")

    try:
        pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = _image_to_tensor(pil, size)

    z = None
    used_model_dtype = "float16" if MODEL_DTYPE == torch.float16 else "float32"

    # Try primary model first
    try:
        with torch.inference_mode():
            t0 = time.time()
            z_try = vae.encode(x).latent_dist.mean * SCALING_FACTOR  # [1,4,H/8,W/8]
            _device_synchronize_if_needed(DEVICE)
            enc_ms = (time.time() - t0) * 1000.0
        if not torch.isfinite(z_try).all():
            raise RuntimeError("non-finite latent in primary encode")
        z = z_try
    except Exception as e:
        # Fallback to FP32 model if available
        print(f"[warn] primary encode failed ({e}); attempting FP32 fallback…")
        if vae_fp32 is None:
            raise HTTPException(status_code=500, detail=f"encode failed and no FP32 fallback available: {e}")
        with torch.inference_mode():
            # Rebuild input as FP32 for fallback
            x32 = x.to(DEVICE, dtype=torch.float32).contiguous(memory_format=torch.channels_last)
            t0 = time.time()
            z_try = vae_fp32.encode(x32).latent_dist.mean * SCALING_FACTOR
            _device_synchronize_if_needed(DEVICE)
            enc_ms = (time.time() - t0) * 1000.0
        if not torch.isfinite(z_try).all():
            raise HTTPException(status_code=500, detail="non-finite latent even in FP32 fallback")
        z = z_try
        used_model_dtype = "float32"

    stats = {
        "min": float(z.float().min()),
        "max": float(z.float().max()),
        "mean": float(z.float().mean()),
        "std": float(z.float().std()),
    }

    shape = list(z.shape)

    # Choose serialization dtype. Prefer configured latent dtype, but fall back to float32 if casting creates non-finite.
    latent_dtype_to_send = LATENT_DTYPE_STR
    try:
        z_for_send = z.detach().to("cpu", dtype=torch.float16 if latent_dtype_to_send == "float16" else torch.float32).contiguous()
        if not np.isfinite(z_for_send.float().numpy()).all():  # type: ignore[attr-defined]
            latent_dtype_to_send = "float32"
    except Exception:
        latent_dtype_to_send = "float32"

    latent_b64 = _encode_latent_to_b64(z, latent_dtype_to_send)

    if save_to_disk and ALLOW_SAVE_TO_DISK:
        pkg = {
            "z": z.detach().to("cpu", dtype=torch.float32).contiguous(),  # save in fp32 for safety
            "scaling": SCALING_FACTOR,
            "size": size,
            "model_id": MODEL_ID,
        }
        ts = int(time.time())
        out_path = os.path.join(SAVE_DIR, f"latent_{ts}.pt")
        torch.save(pkg, out_path)

    return EncodeResponse(
        model_id=MODEL_ID,
        device=str(DEVICE),
        model_dtype=used_model_dtype,
        latent_dtype=latent_dtype_to_send,
        size=size,
        scaling=SCALING_FACTOR,
        shape=shape,
        dtype=latent_dtype_to_send,
        enc_ms=enc_ms,
        stats=stats,
        latent_b64=latent_b64,
    )


@app.post("/decode")
async def decode_latent(req: DecodeRequest = Body(...)) -> Response:
    load_model_once()
    if vae is None:
        raise HTTPException(status_code=500, detail="VAE not initialized")

    scaling = float(req.scaling if req.scaling is not None else (SCALING_FACTOR or 1.0))
    req_dtype = (req.dtype or LATENT_DTYPE_STR).lower()

    with torch.inference_mode():
        z = _decode_b64_to_latent(req.latent_b64, req.shape, req_dtype)
        # Prefer stable FP32 decode on MPS when available
        use_model = vae_fp32 if (DEVICE.type == "mps" and vae_fp32 is not None) else vae
        # Ensure model dtype compatibility
        target_dtype = torch.float32 if use_model is vae_fp32 else MODEL_DTYPE
        z = z.to(DEVICE, dtype=target_dtype).contiguous(memory_format=torch.channels_last)
        t0 = time.time()
        y = use_model.decode(z / scaling).sample  # [-1,1]
        _device_synchronize_if_needed(DEVICE)
        dec_ms = (time.time() - t0) * 1000.0

    img_bytes, media_type = _tensor_to_image_bytes(y, fmt=req.output_format, quality=req.quality, webp_method=req.webp_method)

    # Expose timing for clients that care
    headers = {
        "X-Decode-MS": f"{dec_ms:.2f}",
        "X-Device": str(DEVICE),
        "X-Model": MODEL_ID,
        "X-Model-DType": "float32" if (DEVICE.type == "mps" and vae_fp32 is not None) else ("float16" if MODEL_DTYPE == torch.float16 else "float32"),
    }
    return Response(content=img_bytes, media_type=media_type, headers=headers)


# -------------------------
# Local dev entry
# -------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "vae_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8001")),
        reload=True,
    )