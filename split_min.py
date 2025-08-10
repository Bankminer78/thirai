# split_min.py
import argparse, os, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with images (png/jpg/webp)")
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--val", type=float, default=0.1, help="Val fraction (e.g., 0.1)")
    ap.add_argument("--test", type=float, default=0.0, help="Test fraction (e.g., 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()

    exts = (".png",".jpg",".jpeg",".webp")
    files = [p for p in Path(args.src).glob("*") if p.suffix.lower() in exts]
    if not files:
        raise SystemExit("No images found in --src")

    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    n_test = int(n * args.test)
    n_val  = int(n * args.val)
    n_train = n - n_val - n_test

    splits = {
        "hr_train": files[:n_train],
        "hr_val":   files[n_train:n_train+n_val],
        "hr_test":  files[n_train+n_val:] if n_test > 0 else []
    }

    for name, lst in splits.items():
        if not lst: continue
        outdir = Path(args.out)/name
        outdir.mkdir(parents=True, exist_ok=True)
        for p in lst:
            dst = outdir/p.name
            if args.move:
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))

    print({k: len(v) for k,v in splits.items() if v})

if __name__ == "__main__":
    main()