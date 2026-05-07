from pathlib import Path

from plant_disease.data.manifest import load_manifest


def test_load_manifest_validates_required_columns(tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text("image_path,class_id\nsample.jpg,1\n", encoding="utf-8")

    try:
        load_manifest(csv_path)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected load_manifest to reject incomplete manifest")
