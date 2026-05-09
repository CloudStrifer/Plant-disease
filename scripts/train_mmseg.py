from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _require_mmseg() -> tuple[object, object]:
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "MMSegmentation dependencies are not installed. "
            "Install mmsegmentation, mmengine, and mmcv before running train_mmseg.py."
        ) from exc
    return Config, Runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a standalone MMSegmentation model for lesion segmentation.")
    parser.add_argument("--config", required=True, help="Path to the mmseg config .py file")
    parser.add_argument("--work-dir", default=None, help="Optional override for mmseg work_dir")
    args = parser.parse_args()

    Config, Runner = _require_mmseg()
    import plant_disease_mmseg.dataset  # noqa: F401

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = Config.fromfile(str(config_path))
    if args.work_dir:
        cfg.work_dir = str((ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir))
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

