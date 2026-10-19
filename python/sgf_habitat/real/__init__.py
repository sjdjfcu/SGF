__all__ = [
    "KobukiBaseController",
    "KobukiStatus",
    "RealZedKobukiEnv",
    "ReplaySGFInputEnv",
    "ZEDConfig",
    "ZEDFrame",
    "ZEDFrameProvider",
    "load_sgf_input_dump_metadata",
]


def __getattr__(name: str):
    if name == "RealZedKobukiEnv":
        from .env import RealZedKobukiEnv

        return RealZedKobukiEnv
    if name in {"ReplaySGFInputEnv", "load_sgf_input_dump_metadata"}:
        from .replay import ReplaySGFInputEnv, load_sgf_input_dump_metadata

        return {
            "ReplaySGFInputEnv": ReplaySGFInputEnv,
            "load_sgf_input_dump_metadata": load_sgf_input_dump_metadata,
        }[name]
    if name in {"KobukiBaseController", "KobukiStatus"}:
        from .kobuki import KobukiBaseController, KobukiStatus

        return {"KobukiBaseController": KobukiBaseController, "KobukiStatus": KobukiStatus}[name]
    if name in {"ZEDConfig", "ZEDFrame", "ZEDFrameProvider"}:
        from .zed import ZEDConfig, ZEDFrame, ZEDFrameProvider

        return {"ZEDConfig": ZEDConfig, "ZEDFrame": ZEDFrame, "ZEDFrameProvider": ZEDFrameProvider}[name]
    raise AttributeError(name)
