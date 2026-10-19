#!/usr/bin/env python3
"""Download and validate navigation models before starting episode workers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import sys
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
MOBILECLIP_FILENAME = "mobileclip2_b.ts"
YOLOE_FILENAME = "yoloe-26s-seg.pt"
YOLOE_URL = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{YOLOE_FILENAME}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--task-mode",
        choices=("both", "text", "instance-image"),
        default="both",
        help="Models to initialize: all three (both, default), YOLOE/MobileCLIP2 for text, or RN50 for instance-image.",
    )
    return parser.parse_args()


@contextmanager
def model_lock(model_path: Path) -> Iterator[None]:
    """Serialize initializer processes; navigation workers do not take this lock."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = model_path.with_name(f".{model_path.name}.init.lock")
    with lock_path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for another initializer: {model_path}", flush=True)
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def initialize_rn50() -> Path:
    import clip

    cache_dir = Path.home() / ".cache" / "clip"
    model_path = cache_dir / "RN50.pt"
    with model_lock(model_path):
        print(f"Initializing RN50 (CPU; cached weights are SHA256-checked): {model_path}", flush=True)
        model, preprocess = clip.load(
            "RN50", device="cpu", jit=False, download_root=str(cache_dir)
        )
        del model, preprocess
    return model_path


def initialize_mobileclip2() -> Path:
    import torch
    from ultralytics.utils.downloads import attempt_download_asset

    with model_lock(REPO_ROOT / MOBILECLIP_FILENAME):
        print(f"Initializing MobileCLIP2 (CPU): {MOBILECLIP_FILENAME}", flush=True)
        # Use YOLOE's lookup order: repository cwd, then Ultralytics weights_dir.
        model_path = Path(attempt_download_asset(MOBILECLIP_FILENAME)).resolve()
        try:
            model = torch.jit.load(str(model_path), map_location="cpu").eval()
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load MobileCLIP2 weights at {model_path}: {exc}. "
                "If the cached file is damaged, remove it and rerun this initializer."
            ) from exc
        del model
    return model_path


def initialize_yoloe() -> Path:
    from ultralytics import YOLOE
    from ultralytics.utils.downloads import safe_download

    model_path = REPO_ROOT / "models" / YOLOE_FILENAME
    with model_lock(model_path):
        print(f"Initializing YOLOE (CPU): {model_path}", flush=True)
        if not model_path.is_file():
            safe_download(
                url=YOLOE_URL,
                file=model_path.name,
                dir=model_path.parent,
                unzip=False,
                min_bytes=1e5,
            )
        try:
            model = YOLOE(str(model_path)).to("cpu")
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot load YOLOE weights at {model_path}: {exc}. "
                "If the cached file is damaged, remove it and rerun this initializer."
            ) from exc
        del model
    return model_path


def main() -> int:
    args = parse_args()
    # Batch evaluation also launches navigation subprocesses from REPO_ROOT.
    os.chdir(REPO_ROOT)
    print(f"Python: {sys.executable}", flush=True)
    if args.task_mode in ("both", "instance-image"):
        print(f"Ready: {initialize_rn50()}", flush=True)
    if args.task_mode in ("both", "text"):
        print(f"Ready: {initialize_yoloe()}", flush=True)
        print(f"Ready: {initialize_mobileclip2()}", flush=True)
    print("Model initialization complete. You can now start navigation workers.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. Use the navigation Python environment "
            "and install requirements.txt. For CLIP, install "
            "git+https://github.com/ultralytics/CLIP.git.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Model initialization failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
