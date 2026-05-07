from pathlib import Path

from scripts.train_baseline import resolve_runtime_path


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
