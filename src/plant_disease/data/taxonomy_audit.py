from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PLANT_ALIASES = {
    "corn maize": "corn",
    "maize": "corn",
    "grapevine": "grape",
    "pepper bell": "pepper",
    "bell pepper": "pepper",
}


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("___", " ")
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = value.replace("(", " ")
    value = value.replace(")", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def split_class_name(class_name: str) -> tuple[str, str]:
    if "___" in str(class_name):
        plant, disease = str(class_name).split("___", 1)
        return plant, disease
    return str(class_name), ""


def canonicalize_plant_name(plant_name: str) -> str:
    normalized = normalize_text(plant_name)
    return PLANT_ALIASES.get(normalized, normalized)


def canonicalize_disease_name(plant_name: str, disease_name: str) -> str:
    disease = normalize_text(disease_name)
    plant = canonicalize_plant_name(plant_name)
    disease_tokens = disease.split()
    plant_tokens = plant.split()
    if disease_tokens[: len(plant_tokens)] == plant_tokens:
        disease_tokens = disease_tokens[len(plant_tokens) :]
    return " ".join(disease_tokens) if disease_tokens else disease


def canonical_class_key(class_name: str) -> str:
    plant, disease = split_class_name(class_name)
    return f"{canonicalize_plant_name(plant)}___{canonicalize_disease_name(plant, disease)}"


def build_class_inventory(df: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    if "class_name" not in df.columns:
        raise KeyError("Expected 'class_name' column")
    if source_name is None:
        if "source_dataset" in df.columns and not df["source_dataset"].empty:
            source_name = str(df["source_dataset"].iloc[0])
        else:
            source_name = "unknown"

    rows: list[dict] = []
    for class_name, group in df.groupby("class_name", dropna=True):
        plant, disease = split_class_name(str(class_name))
        rows.append(
            {
                "source_dataset": source_name,
                "class_name": str(class_name),
                "class_id_min": int(group["class_id"].min()) if "class_id" in group else -1,
                "num_samples": int(len(group)),
                "canonical_plant": canonicalize_plant_name(plant),
                "canonical_disease": canonicalize_disease_name(plant, disease),
                "canonical_key": canonical_class_key(str(class_name)),
            }
        )
    return pd.DataFrame(rows).sort_values(["canonical_plant", "canonical_disease", "class_name"]).reset_index(drop=True)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@dataclass
class TaxonomyAuditSummary:
    left_source: str
    right_source: str
    left_num_classes: int
    right_num_classes: int
    exact_name_matches: int
    canonical_matches: int
    ambiguous_matches: int
    left_only: int
    right_only: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def compare_taxonomies(
    left_inventory: pd.DataFrame,
    right_inventory: pd.DataFrame,
    min_ambiguous_score: float = 0.5,
) -> dict[str, pd.DataFrame | TaxonomyAuditSummary]:
    left = left_inventory.copy()
    right = right_inventory.copy()

    exact = left.merge(
        right,
        on="class_name",
        suffixes=("_left", "_right"),
        how="inner",
    )

    canonical = left.merge(
        right,
        on="canonical_key",
        suffixes=("_left", "_right"),
        how="inner",
    )

    canonical_pairs = canonical[["class_name_left", "class_name_right"]].drop_duplicates()
    left_matched = set(canonical_pairs["class_name_left"])
    right_matched = set(canonical_pairs["class_name_right"])

    left_only = left[~left["class_name"].isin(left_matched)].copy()
    right_only = right[~right["class_name"].isin(right_matched)].copy()

    ambiguous_rows: list[dict] = []
    for _, left_row in left_only.iterrows():
        same_plant = right_only[right_only["canonical_plant"] == left_row["canonical_plant"]]
        for _, right_row in same_plant.iterrows():
            score = _token_jaccard(str(left_row["canonical_disease"]), str(right_row["canonical_disease"]))
            if score >= min_ambiguous_score:
                ambiguous_rows.append(
                    {
                        "canonical_plant": left_row["canonical_plant"],
                        "left_class_name": left_row["class_name"],
                        "right_class_name": right_row["class_name"],
                        "left_canonical_disease": left_row["canonical_disease"],
                        "right_canonical_disease": right_row["canonical_disease"],
                        "token_jaccard": score,
                    }
                )

    ambiguous = pd.DataFrame(ambiguous_rows).sort_values(
        ["canonical_plant", "token_jaccard", "left_class_name", "right_class_name"],
        ascending=[True, False, True, True],
    ) if ambiguous_rows else pd.DataFrame(
        columns=[
            "canonical_plant",
            "left_class_name",
            "right_class_name",
            "left_canonical_disease",
            "right_canonical_disease",
            "token_jaccard",
        ]
    )

    summary = TaxonomyAuditSummary(
        left_source=str(left["source_dataset"].iloc[0]) if not left.empty else "left",
        right_source=str(right["source_dataset"].iloc[0]) if not right.empty else "right",
        left_num_classes=int(len(left)),
        right_num_classes=int(len(right)),
        exact_name_matches=int(exact["class_name"].nunique()) if not exact.empty else 0,
        canonical_matches=int(len(canonical_pairs)),
        ambiguous_matches=int(len(ambiguous)),
        left_only=int(len(left_only)),
        right_only=int(len(right_only)),
    )

    return {
        "summary": summary,
        "exact_matches": exact,
        "canonical_matches": canonical,
        "ambiguous_matches": ambiguous,
        "left_only": left_only,
        "right_only": right_only,
    }


def save_audit_report(report: dict[str, pd.DataFrame | TaxonomyAuditSummary], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    assert isinstance(summary, TaxonomyAuditSummary)
    (output_dir / "summary.json").write_text(summary.to_json(), encoding="utf-8")

    for key in ["exact_matches", "canonical_matches", "ambiguous_matches", "left_only", "right_only"]:
        frame = report[key]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(output_dir / f"{key}.csv", index=False)


def build_aligned_class_map(matches_df: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_key", "class_name_left", "class_name_right"}
    missing = required.difference(matches_df.columns)
    if missing:
        raise KeyError(f"matches_df missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    canonical_keys = list(dict.fromkeys(matches_df["canonical_key"].astype(str).tolist()))
    for aligned_class_id, canonical_key in enumerate(canonical_keys):
        subset = matches_df[matches_df["canonical_key"] == canonical_key]
        rows.append(
            {
                "aligned_class_id": aligned_class_id,
                "aligned_class_name": canonical_key,
                "canonical_key": canonical_key,
                "left_class_name": str(subset["class_name_left"].iloc[0]),
                "right_class_name": str(subset["class_name_right"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_aligned_manifest(manifest_df: pd.DataFrame, class_map: pd.DataFrame, side: str) -> pd.DataFrame:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    class_name_column = f"{side}_class_name"
    if class_name_column not in class_map.columns:
        raise KeyError(f"class_map missing required column: {class_name_column}")
    if "class_name" not in manifest_df.columns:
        raise KeyError("manifest_df missing required column: class_name")

    mapping = class_map[[class_name_column, "aligned_class_id", "aligned_class_name", "canonical_key"]].copy()
    mapping = mapping.rename(columns={class_name_column: "class_name"})
    aligned = manifest_df.merge(mapping, on="class_name", how="inner")

    if aligned.empty:
        return aligned

    aligned = aligned.copy()
    aligned["original_class_id"] = aligned["class_id"]
    aligned["original_class_name"] = aligned["class_name"]
    aligned["class_id"] = aligned["aligned_class_id"].astype(int)
    aligned["class_name"] = aligned["aligned_class_name"].astype(str)

    keep_columns = list(manifest_df.columns) + [
        "original_class_id",
        "original_class_name",
        "canonical_key",
        "aligned_class_id",
        "aligned_class_name",
    ]
    return aligned[keep_columns]


def build_aligned_manifests(
    left_manifest_df: pd.DataFrame,
    right_manifest_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    class_map = build_aligned_class_map(matches_df)
    left_aligned = build_aligned_manifest(left_manifest_df, class_map, side="left")
    right_aligned = build_aligned_manifest(right_manifest_df, class_map, side="right")
    mixed_aligned = pd.concat([left_aligned, right_aligned], ignore_index=True)
    return {
        "class_map": class_map,
        "left_aligned": left_aligned,
        "right_aligned": right_aligned,
        "mixed_aligned": mixed_aligned,
    }


def save_aligned_manifests(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs["class_map"].to_csv(output_dir / "aligned_class_map.csv", index=False)
    outputs["left_aligned"].to_csv(output_dir / "left_aligned_manifest.csv", index=False)
    outputs["right_aligned"].to_csv(output_dir / "right_aligned_manifest.csv", index=False)
    outputs["mixed_aligned"].to_csv(output_dir / "mixed_aligned_manifest.csv", index=False)
