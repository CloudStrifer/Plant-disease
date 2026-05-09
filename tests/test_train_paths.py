from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_model import _build_model as build_eval_model
from scripts.train_baseline import build_model, resolve_model_in_channels, resolve_runtime_path


def test_resolve_runtime_path_uses_repo_root_for_repo_relative_values(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "baseline.yaml"
    config_path.write_text("x: 1", encoding="utf-8")

    resolved = resolve_runtime_path("data/manifests/plantseg.csv", config_path=config_path, repo_root=repo_root)

    assert resolved == repo_root / "data" / "manifests" / "plantseg.csv"


def test_resolve_runtime_path_keeps_absolute_values(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs" / "baseline.yaml"
    absolute = tmp_path / "other" / "manifest.csv"

    resolved = resolve_runtime_path(str(absolute), config_path=config_path, repo_root=repo_root)

    assert resolved == absolute


def test_train_build_model_uses_backbone_specific_default_in_channels() -> None:
    df = pd.DataFrame(
        {
            "class_id": [0, 1],
            "severity_label": [0, 1],
        }
    )

    model = build_model(df, {"backbone_name": "segformer_b0"})

    assert model.seg_head.in_channels == 256
    assert model.sev_head.fc.in_features == 256


def test_train_build_model_allows_in_channels_override() -> None:
    df = pd.DataFrame(
        {
            "class_id": [0, 1],
            "severity_label": [0, 1],
        }
    )

    model = build_model(df, {"backbone_name": "segformer_b0", "in_channels": 320})

    assert model.seg_head.in_channels == 320
    assert model.sev_head.fc.in_features == 320


def test_eval_build_model_uses_backbone_specific_default_in_channels() -> None:
    model = build_eval_model({"backbone_name": "segformer_b0"}, num_classes=3)

    assert model.seg_head.in_channels == 256
    assert model.sev_head.fc.in_features == 256


def test_eval_build_model_allows_in_channels_override() -> None:
    model = build_eval_model({"backbone_name": "segformer_b0", "in_channels": 320}, num_classes=3)

    assert model.seg_head.in_channels == 320
    assert model.sev_head.fc.in_features == 320


def test_resolve_model_in_channels_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="Unsupported backbone_name"):
        resolve_model_in_channels({"backbone_name": "typoformer_b0"})
