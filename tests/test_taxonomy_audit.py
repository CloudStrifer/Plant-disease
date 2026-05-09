from pathlib import Path

import pandas as pd

from plant_disease.data.taxonomy_audit import (
    build_aligned_manifests,
    build_class_inventory,
    canonical_class_key,
    compare_taxonomies,
    save_aligned_manifests,
    save_audit_report,
)


def test_canonical_class_key_normalizes_common_cross_dataset_variants() -> None:
    assert canonical_class_key("Apple___apple black rot") == "apple___black rot"
    assert canonical_class_key("Apple___Black_rot") == "apple___black rot"
    assert canonical_class_key("Corn_(maize)___Northern_Leaf_Blight") == "corn___northern leaf blight"
    assert canonical_class_key("Corn___corn northern leaf blight") == "corn___northern leaf blight"


def test_compare_taxonomies_finds_canonical_matches_and_ambiguities(tmp_path: Path) -> None:
    left_df = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "Apple___apple black rot", "source_dataset": "PlantSeg"},
            {"class_id": 1, "class_name": "Tomato___tomato leaf mold", "source_dataset": "PlantSeg"},
            {"class_id": 2, "class_name": "Grape___grape downy mildew", "source_dataset": "PlantSeg"},
        ]
    )
    right_df = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "Apple___Black_rot", "source_dataset": "PlantVillage"},
            {"class_id": 1, "class_name": "Tomato___Leaf_Mold", "source_dataset": "PlantVillage"},
            {"class_id": 2, "class_name": "Tomato___Bacterial_spot", "source_dataset": "PlantVillage"},
        ]
    )
    left_df.loc[len(left_df)] = {
        "class_id": 3,
        "class_name": "Tomato___tomato bacterial leaf spot",
        "source_dataset": "PlantSeg",
    }

    report = compare_taxonomies(build_class_inventory(left_df), build_class_inventory(right_df), min_ambiguous_score=0.25)

    canonical = report["canonical_matches"]
    ambiguous = report["ambiguous_matches"]
    summary = report["summary"]

    assert len(canonical) == 2
    assert len(ambiguous) >= 1
    assert summary.canonical_matches == 2

    save_audit_report(report, tmp_path / "audit")
    assert (tmp_path / "audit" / "summary.json").exists()


def test_build_aligned_manifests_reindexes_shared_classes(tmp_path: Path) -> None:
    left_manifest = pd.DataFrame(
        [
            {
                "image_path": "left_a.png",
                "class_id": 10,
                "class_name": "Apple___apple black rot",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantSeg",
                "severity_label": 0,
            },
            {
                "image_path": "left_b.png",
                "class_id": 99,
                "class_name": "Banana___banana bunchy top",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantSeg",
                "severity_label": 0,
            },
        ]
    )
    right_manifest = pd.DataFrame(
        [
            {
                "image_path": "right_a.png",
                "class_id": 3,
                "class_name": "Apple___Black_rot",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "pseudo_label": True,
            },
            {
                "image_path": "right_b.png",
                "class_id": 8,
                "class_name": "Tomato___Leaf_Mold",
                "has_lesion_mask": True,
                "has_leaf_mask": False,
                "source_dataset": "PlantVillage_pseudo",
                "severity_label": 0,
                "pseudo_label": True,
            },
        ]
    )
    matches = pd.DataFrame(
        [
            {
                "canonical_key": "apple___black rot",
                "class_name_left": "Apple___apple black rot",
                "class_name_right": "Apple___Black_rot",
            },
            {
                "canonical_key": "tomato___leaf mold",
                "class_name_left": "Tomato___tomato leaf mold",
                "class_name_right": "Tomato___Leaf_Mold",
            },
        ]
    )

    outputs = build_aligned_manifests(left_manifest, right_manifest, matches)
    assert list(outputs["class_map"]["aligned_class_id"]) == [0, 1]
    assert len(outputs["left_aligned"]) == 1
    assert len(outputs["right_aligned"]) == 2
    assert outputs["left_aligned"].iloc[0]["class_id"] == 0
    assert outputs["right_aligned"].iloc[0]["class_name"] == "apple___black rot"

    save_aligned_manifests(outputs, tmp_path / "aligned")
    assert (tmp_path / "aligned" / "mixed_aligned_manifest.csv").exists()
