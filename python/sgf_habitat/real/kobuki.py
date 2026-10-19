from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BRIDGE = REPO_ROOT / "bin" / "sgf-kobuki-json-bridge"


@dataclass
class KobukiStatus:
    alive: bool = False
    enabled: bool = False
    bumper: int = 0
    cliff: int = 0
    wheel_drop: int = 0
    battery: int = 0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    safety_stop: bool = False
    velocity_timeout_supported: bool = False
    raw: dict[str, Any] | None = None


class KobukiBaseController:
    def __init__(
        self,
        *,
        device: str = "/dev/kobuki",
        kobuki_install: str | Path = "~/kobuki/install",
        bridge_binary: str | Path = DEFAULT_BRIDGE,
        startup_timeout_s: float = 6.0,
    ) -> None:
        self.device = str(device)
        self.kobuki_install = Path(kobuki_install).expanduser()
        self.bridge_binary = Path(bridge_binary).expanduser()
        self.startup_timeout_s = float(startup_timeout_s)
        self._process: subprocess.Popen[str] | None = None
        self._status_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_lines: queue.Queue[str] = queue.Queue()
        self._reader_threads: list[threading.Thread] = []
        self._latest_status = KobukiStatus()
        self._last_status_received_at: float | None = None

    @property
    def latest_status(self) -> KobukiStatus:
        self._drain_status()
        return self._latest_status

    @property
    def status_age_s(self) -> float:
        received = self._last_status_received_at
        return float("inf") if received is None else time.monotonic() - received

    def start(self) -> None:
        if self._process is not None:
            return
        if not self.bridge_binary.exists():
            raise FileNotFoundError(
                f"Kobuki bridge binary not found: {self.bridge_binary}. "
                "Build it with `python3 tools/build_kobuki_bridge.py --kobuki-install ~/kobuki/install`."
            )
        env = self._bridge_environment()
        self._process = subprocess.Popen(
            [str(self.bridge_binary), self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(REPO_ROOT),
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._reader_threads = [
            threading.Thread(target=self._stdout_reader, args=(self._process.stdout,), daemon=True),
            threading.Thread(target=self._stderr_reader, args=(self._process.stderr,), daemon=True),
        ]
        for thread in self._reader_threads:
            thread.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._collect_stderr()
                raise RuntimeError(f"Kobuki bridge exited during startup: {stderr}")
            self._drain_status()
            if self._latest_status.alive or (self._latest_status.raw or {}).get("event") == "ready":
                return
            time.sleep(0.05)
        raise TimeoutError("timed out waiting for Kobuki bridge startup")

    def set_velocity(self, linear_mps: float, angular_radps: float, *, timeout_s: float | None = None) -> None:
        payload = {"cmd": "velocity", "linear": float(linear_mps), "angular": float(angular_radps)}
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                raise ValueError("velocity timeout must be finite and positive")
            payload["timeout_s"] = float(timeout_s)
        self._send(payload)

    def stop(self) -> None:
        self._send({"cmd": "stop"})

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Kobuki bridge is not running")
        if process.poll() is not None:
            stderr = self._collect_stderr()
            raise RuntimeError(f"Kobuki bridge is not running: {stderr}")
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _bridge_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        lib_dir = self.kobuki_install / "lib"
        bin_dir = self.kobuki_install / "bin"
        share_dir = self.kobuki_install / "share"
        env["LD_LIBRARY_PATH"] = self._prepend_env_path("LD_LIBRARY_PATH", lib_dir, env)
        env["PATH"] = self._prepend_env_path("PATH", bin_dir, env)
        env["CMAKE_PREFIX_PATH"] = self._prepend_env_path("CMAKE_PREFIX_PATH", self.kobuki_install, env)
        env["AMENT_PREFIX_PATH"] = self._prepend_env_path("AMENT_PREFIX_PATH", self.kobuki_install, env)
        env["COLCON_PREFIX_PATH"] = self._prepend_env_path("COLCON_PREFIX_PATH", self.kobuki_install, env)
        env["PKG_CONFIG_PATH"] = self._prepend_env_path("PKG_CONFIG_PATH", share_dir / "pkgconfig", env)
        return env

    @staticmethod
    def _prepend_env_path(name: str, path: Path, env: dict[str, str]) -> str:
        current = env.get(name, "")
        path_text = str(path)
        if not current:
            return path_text
        return path_text + os.pathsep + current

    def _stdout_reader(self, stream) -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if payload.get("event") == "status":
                    self._last_status_received_at = time.monotonic()
                self._status_queue.put(payload)
            except json.JSONDecodeError:
                self._status_queue.put({"event": "log", "message": line})

    def _stderr_reader(self, stream) -> None:
        for line in stream:
            line = line.strip()
            if line:
                self._stderr_lines.put(line)

    def _drain_status(self) -> None:
        while True:
            try:
                raw = self._status_queue.get_nowait()
            except queue.Empty:
                return
            self._latest_status = self._status_from_raw(raw)

    def _status_from_raw(self, raw: dict[str, Any]) -> KobukiStatus:
        return KobukiStatus(
            alive=bool(raw.get("alive", self._latest_status.alive)),
            enabled=bool(raw.get("enabled", self._latest_status.enabled)),
            bumper=int(raw.get("bumper", self._latest_status.bumper)),
            cliff=int(raw.get("cliff", self._latest_status.cliff)),
            wheel_drop=int(raw.get("wheel_drop", self._latest_status.wheel_drop)),
            battery=int(raw.get("battery", self._latest_status.battery)),
            x=float(raw.get("x", self._latest_status.x)),
            y=float(raw.get("y", self._latest_status.y)),
            heading=float(raw.get("heading", self._latest_status.heading)),
            safety_stop=bool(raw.get("safety_stop", self._latest_status.safety_stop)),
            velocity_timeout_supported=bool(raw.get("velocity_timeout_supported", self._latest_status.velocity_timeout_supported)),
            raw=dict(raw),
        )

    def _collect_stderr(self) -> str:
        lines: list[str] = []
        while True:
            try:
                lines.append(self._stderr_lines.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines[-20:])
