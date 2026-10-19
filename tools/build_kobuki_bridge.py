#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the direct Kobuki JSON-lines bridge.")
    parser.add_argument("--build-dir", type=Path, default=Path("build/kobuki_bridge"))
    parser.add_argument("--kobuki-install", type=Path, default=Path("~/kobuki/install"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    kobuki_install = args.kobuki_install.expanduser().resolve()
    configure = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DBUILD_KOBUKI_BRIDGE=ON",
        "-DBUILD_PYTHON=OFF",
        f"-DKOBUKI_INSTALL_PREFIX={kobuki_install}",
        f"-DCMAKE_PREFIX_PATH={kobuki_install}",
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
    ]
    subprocess.run(configure, check=True, cwd=repo_root)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "sgf-kobuki-json-bridge", "-j"],
        check=True,
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
