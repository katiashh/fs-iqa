"""DBCNN evaluation entry point. Run `python -m fs_iqa.cli --help`."""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

from .data import load_mos, select_images


def parser():
    p = argparse.ArgumentParser(description="FS-IQA DBCNN inference and empirical IVE")
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--image-list", type=Path, help="One path relative to image-dir per line")
    p.add_argument("--mos-csv", type=Path)
    p.add_argument("--mos-column", default="MOS_zscore")
    p.add_argument("--backbone-checkpoint", type=Path,
                   default=Path("checkpoints/DBCNN_KonIQ10k-2de81c0a.pth"))
    p.add_argument("--ftn-checkpoint", type=Path,
                   default=Path("checkpoints/dbcnn_prevscorer_FS_01.pth"))
    p.add_argument("--scorer-checkpoint", type=Path,
                   default=Path("checkpoints/dbcnn_scorer_FS_01.pth"))
    p.add_argument("--mode", choices=("score", "interval", "ive"), default="score",
                   help="ive adds the empirical input-radius proxy; it is expensive")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--smooth-batch-size", type=int, default=100)
    p.add_argument("--confidence", type=float, default=0.999)
    p.add_argument("--hessian-samples", type=int, default=100)
    p.add_argument("--fd-step", type=float, default=1e-3)
    p.add_argument("--jacobian-threshold", type=float, default=1e-3)
    p.add_argument("--no-feature-clipping", action="store_true",
                   help="Disable the clipping used in the supplied demo")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/dbcnn.csv"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate image selection and report missing files without loading models")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.output.suffix.lower() != ".csv":
        p.error("--output must have a .csv extension")
    for name in ("sigma", "fd_step"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("kappa", "jacobian_threshold"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("samples", "smooth_batch_size", "hessian_samples", "width", "height"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.confidence < 1:
        p.error("--confidence must be between 0 and 1")
    try:
        images = select_images(args.image_dir, image_list=args.image_list,
                               start_index=args.start_index, shuffle=args.shuffle,
                               seed=args.seed, limit=args.limit)
        mos = load_mos(args.mos_csv, args.mos_column)
        if args.mos_csv:
            missing_mos = [im.name for im in images if im.name not in mos]
            if missing_mos:
                raise ValueError(f"Missing MOS for {len(missing_mos)} images, e.g. {missing_mos[0]}")
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    checkpoint_paths = [args.backbone_checkpoint, args.ftn_checkpoint, args.scorer_checkpoint]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if args.dry_run:
        print(json.dumps({"selected_images": len(images), "first_image": str(images[0]),
                          "width": args.width, "height": args.height, "mode": args.mode,
                          "feature_radius": args.kappa * args.sigma,
                          "missing_checkpoints": missing}, indent=2))
        return 0
    if missing:
        p.error("Missing checkpoints: " + ", ".join(missing) + ". See checkpoints/README.md")
    if not args.overwrite:
        for out in (args.output, args.output.with_suffix(".json")):
            if out.exists():
                p.error(f"Output exists: {out}; choose another name or pass --overwrite")

    import numpy as np
    import torch
    import torchvision
    from torch import nn
    from tqdm import tqdm

    from . import __version__
    from .data import read_image
    from .ive import estimate_input_radius
    from .models import load_model
    from .smoothing import predict

    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is unavailable")
    print(f"Loading DBCNN and FS-IQA heads on {device}. The FTN alone needs about 10 GiB in float32.")
    model = load_model(*checkpoint_paths, device=device)
    feature_map = nn.Sequential(model.backbone, model.ftn).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = [str(im.relative_to(args.image_dir.resolve())) for im in images]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {"config": config, "version": __version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "device": device,
                "feature_radius": args.kappa * args.sigma,
                "input_radius_kind": "empirical_proxy" if args.mode == "ive" else "not_requested",
                "selected_images": selected,
                "selection_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
                "run_status": "running", "completed_images": 0}
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    fields = ["image_name", "median", "lower_b", "upper_b", "gt", "epsilon_x_estimate",
              "jacobian_norm", "curvature_proxy", "interval_status", "ive_status",
              "n_samples", "seconds"]
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for path, image_name in tqdm(zip(images, selected), total=len(images)):
                x = read_image(path, width=args.width, height=args.height).to(device)
                # Stable per-image noise regardless of batch order or IVE mode.
                image_seed = int.from_bytes(hashlib.sha256(
                    f"{args.seed}:{image_name}".encode()).digest()[:8], "little") % (2**63 - 1)
                generator = torch.Generator(device=device).manual_seed(image_seed)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    features = model.features(x)
                row = predict(features, model.scorer, sigma=args.sigma, kappa=args.kappa,
                              n=args.samples, batch_size=args.smooth_batch_size,
                              confidence=args.confidence, score_only=args.mode == "score",
                              clip_features=not args.no_feature_clipping, generator=generator)
                row.update({"ive_status": "not_requested", "image_name": image_name,
                            "gt": mos.get(path.name)})
                if args.mode == "ive":
                    row.update(estimate_input_radius(feature_map, x,
                               feature_radius=args.kappa * args.sigma,
                               n_hessian=args.hessian_samples, fd_step=args.fd_step,
                               jacobian_threshold=args.jacobian_threshold, generator=generator))
                if device == "cuda":
                    torch.cuda.synchronize()
                row["seconds"] = time.perf_counter() - start
                writer.writerow(row)
                handle.flush()
                metadata["completed_images"] += 1
        metadata["run_status"] = "complete"
    finally:
        if metadata["run_status"] != "complete":
            metadata["run_status"] = "incomplete"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output} and {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
