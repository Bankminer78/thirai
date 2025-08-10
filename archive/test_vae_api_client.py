from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test client for SDXL VAE FastAPI encode/decode endpoints")
    parser.add_argument("--host", type=str, default="http://localhost:8001", help="Base URL of the API server")
    parser.add_argument("--image", type=str, default=str(Path(__file__).parent / "crop.jpg"), help="Path to input image")
    parser.add_argument("--size", type=int, default=384, help="Resize square size before encode")
    parser.add_argument("--format", type=str, default="webp", choices=["webp", "png"], help="Decoded image format")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "recon.webp"), help="Path to save decoded image")
    args = parser.parse_args()

    base_url: str = args.host.rstrip("/")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # --- encode ---
    encode_url = f"{base_url}/encode?size={args.size}"
    mime_guess, _ = mimetypes.guess_type(str(image_path))
    if mime_guess is None:
        mime_guess = "application/octet-stream"

    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime_guess)}
        enc_res = requests.post(encode_url, files=files, timeout=120)
    enc_res.raise_for_status()

    enc_json: Dict[str, Any] = enc_res.json()
    latent_b64: str = enc_json["latent_b64"]
    shape: List[int] = enc_json["shape"]
    scaling: float = float(enc_json["scaling"])
    enc_ms: float = float(enc_json.get("enc_ms", 0.0))
    stats = enc_json.get("stats", {})

    print(f"[encode] {enc_ms:.2f} ms | device={enc_json.get('device')} | size={enc_json.get('size')}")
    print(f"[latent] shape={shape} dtype=float32 scaling={scaling}")
    print(f"[latent] stats: min={stats.get('min'):.4f} max={stats.get('max'):.4f} mean={stats.get('mean'):.4f} std={stats.get('std'):.4f}")

    # --- decode ---
    decode_url = f"{base_url}/decode"
    body = {
        "latent_b64": latent_b64,
        "shape": shape,
        "scaling": scaling,
        "output_format": args.format,
    }
    dec_res = requests.post(decode_url, headers={"Content-Type": "application/json"}, data=json.dumps(body), timeout=120)
    dec_res.raise_for_status()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(dec_res.content)

    dec_ms_header = dec_res.headers.get("X-Decode-MS", "")
    device_header = dec_res.headers.get("X-Device", "")
    print(f"[decode] {dec_ms_header} ms | device={device_header} | saved -> {out_path}")


if __name__ == "__main__":
    main()