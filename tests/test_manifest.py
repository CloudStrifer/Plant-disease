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


def test_load_manifest_attaches_manifest_directory(tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "\n".join(
            [
                "image_path,class_id,has_lesion_mask,has_leaf_mask,source_dataset,severity_label",
                "sample.jpg,1,False,False,unit,0",
            ]
        ),
        encoding="utf-8",
    )

    df = load_manifest(csv_path)

    assert "__manifest_dir" in df.columns
    assert df.loc[0, "__manifest_dir"] == str(tmp_path.resolve())
