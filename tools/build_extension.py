#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SGF pybind11 extension.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/python"),
        help="Out-of-source CMake build directory.",
    )
    parser.add_argument("--debug", action="store_true", help="Build with Debug configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = (repo_root / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    pybind11_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        text=True,
        cwd=repo_root,
    ).strip()

    cmake_configure = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DPython3_EXECUTABLE={Path(sys.executable).resolve()}",
        f"-DPython3_ROOT_DIR={Path(sys.prefix).resolve()}",
        "-DPython3_FIND_STRATEGY=LOCATION",
        "-DBUILD_PYTHON=ON",
        f"-DCMAKE_BUILD_TYPE={'Debug' if args.debug else 'Release'}",
    ]
    subprocess.run(cmake_configure, check=True, cwd=repo_root)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "sgf_habitat_core", "-j"],
        check=True,
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
