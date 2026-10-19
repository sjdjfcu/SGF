#!/usr/bin/env python3

import argparse
import contextlib
import io
import json
import os
import struct
import sys
import time
import traceback
import warnings


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)
DEFAULT_EPISODE_CPU_THREADS = 1


def _parse_positive_int(value: str | None, default: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _configured_cpu_threads() -> int:
    explicit = os.environ.get("SGF_EPISODE_CPU_THREADS")
    if explicit is not None:
        return _parse_positive_int(explicit, default=0)
    inherited = os.environ.get("OMP_NUM_THREADS")
    if inherited is not None:
        return _parse_positive_int(inherited, default=0)
    return DEFAULT_EPISODE_CPU_THREADS


def _apply_thread_env_defaults() -> None:
    threads = _configured_cpu_threads()
    if threads <= 0:
        return
    value = str(threads)
    os.environ.setdefault("SGF_EPISODE_CPU_THREADS", value)
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, value)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_apply_thread_env_defaults()

import numpy as np
import torch


def _apply_runtime_thread_limits() -> None:
    threads = _configured_cpu_threads()
    if threads <= 0:
        return
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    interop_threads = _parse_positive_int(
        os.environ.get("SGF_EPISODE_TORCH_INTEROP_THREADS"),
        default=1,
    )
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


_apply_runtime_thread_limits()

from ultralytics import YOLOE

warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)

REQUEST_HEADER = struct.Struct("<IIIffII")
RESPONSE_HEADER = struct.Struct("<III")
DETECTION_HEADER = struct.Struct("<ifiiiiiII")
MESSAGE_SIZE = struct.Struct("<Q")


class ClassConfigurationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classes", action="append", default=[])
    return parser.parse_args()


def read_exact(reader, size: int):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = reader.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(reader):
    size_bytes = read_exact(reader, MESSAGE_SIZE.size)
    if size_bytes is None:
        return None
    (size,) = MESSAGE_SIZE.unpack(size_bytes)
    if size == 0:
        return b""
    return read_exact(reader, int(size))


def write_message(writer, payload: bytes):
    writer.write(MESSAGE_SIZE.pack(len(payload)))
    if payload:
        writer.write(payload)
    writer.flush()


def pack_error_response(message: str):
    error_bytes = message.encode("utf-8", errors="replace")
    return RESPONSE_HEADER.pack(0, len(error_bytes), 0) + error_bytes


def pack_success_response(detections):
    payload = bytearray()
    payload += RESPONSE_HEADER.pack(1, 0, len(detections))
    for detection in detections:
        mask = detection.get("mask")
        if mask is None:
            mask_rows = 0
            mask_cols = 0
            mask_size = 0
            mask_bytes = b""
        else:
            mask_rows = int(mask.shape[0])
            mask_cols = int(mask.shape[1])
            mask_size = int(mask.size)
            mask_bytes = mask.tobytes()
        payload += DETECTION_HEADER.pack(
            int(detection["class_id"]),
            float(detection["confidence"]),
            int(detection["bbox"][0]),
            int(detection["bbox"][1]),
            int(detection["bbox"][2]),
            int(detection["bbox"][3]),
            mask_rows,
            mask_cols,
            mask_size,
        )
        payload += mask_bytes
    return bytes(payload)


def choose_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def configured_class_names(model: YOLOE) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        ordered = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        ordered = list(names)
    return [str(name) for name in ordered]


def validate_configured_classes(requested: list[str], actual: list[str]) -> None:
    if (
        any(not name for name in requested)
        or any(not name for name in actual)
        or len(set(requested)) != len(requested)
        or len(set(actual)) != len(actual)
        or len(actual) != len(requested)
        or set(actual) != set(requested)
    ):
        raise ClassConfigurationError(
            "YOLOE configured classes do not match requested queries: "
            f"requested={requested!r}, actual={actual!r}"
        )


def initialize_model(args):
    last_error = ""
    for attempt in range(3):
        model = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = YOLOE(args.model)
            model.to(choose_device())
            if args.classes:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        model.set_classes(args.classes)
                except Exception as exc:
                    raise ClassConfigurationError(
                        f"YOLOE set_classes failed for {args.classes!r}: {exc}"
                    ) from exc
                validate_configured_classes(
                    list(args.classes),
                    configured_class_names(model),
                )
            return model, ""
        except ClassConfigurationError:
            last_error = traceback.format_exc()
            break
        except Exception:
            last_error = traceback.format_exc()
            try:
                del model
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(10.0 * float(attempt + 1))
    return None, last_error


def handle_request(model: YOLOE, payload: bytes):
    if payload is None or len(payload) < REQUEST_HEADER.size:
        return pack_error_response("invalid request payload")

    width, height, channels, conf_threshold, iou_threshold, request_masks, image_bytes = REQUEST_HEADER.unpack_from(payload, 0)
    image_offset = REQUEST_HEADER.size
    if len(payload) != image_offset + image_bytes:
        return pack_error_response("request payload size mismatch")
    if channels != 3:
        return pack_error_response(f"unsupported channel count: {channels}")

    image = np.frombuffer(payload, dtype=np.uint8, count=image_bytes, offset=image_offset)
    image = image.reshape((height, width, channels))

    with contextlib.redirect_stdout(io.StringIO()):
        results = model.predict(
            source=image,
            verbose=False,
            conf=float(conf_threshold),
            iou=float(iou_threshold),
            imgsz=640,
            retina_masks=bool(request_masks),
        )

    detections = []
    for result in results:
        if result.boxes is None:
            continue

        masks = None
        if request_masks and result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

        for idx in range(len(classes)):
            bbox = boxes[idx].astype(np.int32).tolist()
            detections.append(
                {
                    "class_id": int(classes[idx]),
                    "confidence": float(confs[idx]),
                    "bbox": [
                        int(bbox[0]),
                        int(bbox[1]),
                        int(max(0, bbox[2] - bbox[0])),
                        int(max(0, bbox[3] - bbox[1])),
                    ],
                    "mask": None if masks is None else (masks[idx] > 0.5).astype(np.uint8) * 255,
                }
            )

    return pack_success_response(detections)


def main():
    args = parse_args()
    model, init_error = initialize_model(args)
    if model is None:
        ready_payload = json.dumps({"ok": False, "error": init_error or "YOLO bridge initialization failed"}).encode(
            "utf-8"
        )
        write_message(sys.stdout.buffer, ready_payload)
        return

    names = model.names if isinstance(model.names, dict) else {idx: name for idx, name in enumerate(model.names)}
    ready_payload = json.dumps(
        {
            "ok": True,
            "names": {str(int(idx)): str(name) for idx, name in names.items()},
        }
    ).encode("utf-8")
    write_message(sys.stdout.buffer, ready_payload)

    while True:
        payload = read_message(sys.stdin.buffer)
        if payload is None:
            break
        try:
            response = handle_request(model, payload)
        except Exception as exc:  # pragma: no cover - best effort bridge
            response = pack_error_response(f"{exc}\n{traceback.format_exc()}")
        write_message(sys.stdout.buffer, response)


if __name__ == "__main__":
    main()
