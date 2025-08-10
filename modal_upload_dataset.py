# modal_upload_dataset.py
import argparse, os
from pathlib import Path
import modal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="ae-data", help="Modal Volume name to create/use")
    ap.add_argument("--local_dir", required=True, help="Local dataset root")
    ap.add_argument("--remote_dir", default="/datasets/myface_ds", help="Remote path inside the Volume")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files in the Volume if they exist")
    args = ap.parse_args()

    src = Path(args.local_dir)
    if not src.exists():
        raise SystemExit(f"Local dir not found: {src}")

    # Expectation: src contains hr_train/ and hr_val/ (384x384 crops)
    need = [src/"hr_train", src/"hr_val"]
    for p in need:
        if not p.exists():
            raise SystemExit(f"Missing subfolder: {p} (expected hr_train/ and hr_val/)")

    vol = modal.Volume.from_name(args.volume, create_if_missing=True)
    print(f"Uploading {src} -> {args.remote_dir} in Volume '{args.volume}' ...")
    with vol.batch_upload(force=args.force) as batch:
        batch.put_directory(str(src), args.remote_dir)
    print("Done! (committed)")

if __name__ == "__main__":
    main()