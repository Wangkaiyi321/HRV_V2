"""HRV time-domain analysis from preprocessed RRI/NNI files.

This program deliberately does *not* filter ECG, detect R peaks, flag
artifacts, or correct intervals.  Those operations belong to
``hrv_preprocess.py``.  It reads each ``rri_and_nni.csv`` below an input
folder and produces auditable NNI, block-level, condition-level, and group
summary tables.

Example (the three arguments below are researcher decisions)::

    python hrv_time_domain.py --input-dir hrv_output --sdnn-ddof 1 \
        --ci-method t --missing-block-policy complete_blocks

No significance tests, p values, multiple-comparison corrections, effect
sizes, or figures are implemented here.  They require a separate approved
statistical/figure specification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_NNI_COLUMNS = {
    "source_file", "participant", "segment", "condition",
    "nni_corrected_ms", "correction_action", "review_status",
}
REST_SEGMENT = r"^(pre|post)_rest_\d+_(EC|EO)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("hrv_output"), help="Folder containing rri_and_nni.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("hrv_time_domain_output"), help="Folder for analysis outputs.")
    parser.add_argument("--sdnn-ddof", required=True, choices=("0", "1"), help="Researcher-approved SDNN estimator: 0=population, 1=sample.")
    parser.add_argument("--ci-method", required=True, choices=("t",), help="Researcher-approved 95%% CI estimator. Only t-based CI is currently implemented.")
    parser.add_argument("--missing-block-policy", required=True, choices=("complete_blocks", "available_blocks"), help="Researcher-approved handling for missing resting blocks.")
    return parser.parse_args()


def discover_inputs(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.rglob("rri_and_nni.csv"))
    if not paths:
        raise ValueError(f"INPUT_ERROR: no rri_and_nni.csv files found under {input_dir.resolve()}")
    return paths


def load_nni(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_NNI_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"SCHEMA_ERROR: {path} lacks required columns: {', '.join(sorted(missing))}")
    frame["input_path"] = str(path)
    return frame


def load_qc_status(nni_path: Path) -> pd.DataFrame:
    """Load segment QC when present; absence is explicit rather than guessed."""
    qc_path = nni_path.with_name("preprocessing_qc.csv")
    columns = ["source_file", "participant", "segment", "qc_status", "qc_exclusion_reason"]
    if not qc_path.exists():
        return pd.DataFrame(columns=columns)
    qc = pd.read_csv(qc_path, encoding="utf-8-sig")
    required = {"source_file", "participant", "segment", "status", "exclusion_reason"}
    missing = required - set(qc.columns)
    if missing:
        raise ValueError(f"SCHEMA_ERROR: {qc_path} lacks required columns: {', '.join(sorted(missing))}")
    return qc[["source_file", "participant", "segment", "status", "exclusion_reason"]].rename(
        columns={"status": "qc_status", "exclusion_reason": "qc_exclusion_reason"}
    )


def add_hierarchy(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive analysis hierarchy without inferring any unapproved state labels."""
    values = frame.copy()
    parsed = values["segment"].astype(str).str.extract(REST_SEGMENT, expand=True)
    values["Time"] = parsed[0].map({"pre": "Pre", "post": "Post"})
    values["Condition"] = parsed[1]
    values["Block"] = pd.NA

    rest_mask = values["Time"].notna()
    # Count unique segments, never individual NNi rows, within each EC/EO state.
    segment_blocks = (
        values.loc[rest_mask, ["participant", "source_file", "Time", "Condition", "segment"]]
        .drop_duplicates()
        .sort_values(["participant", "source_file", "Time", "Condition", "segment"])
    )
    segment_blocks["Block"] = (
        segment_blocks.groupby(["participant", "source_file", "Time", "Condition"], sort=False)
        .cumcount().add(1).astype(str).radd(segment_blocks["Condition"].astype(str))
    )
    values = values.merge(segment_blocks, how="left", on=["participant", "source_file", "Time", "Condition", "segment"], suffixes=("", "_derived"), validate="many_to_one")
    values["Block"] = values.pop("Block_derived").combine_first(values["Block"])
    task_mask = values["segment"].astype(str).str.casefold().eq("task")
    values.loc[task_mask, ["Time", "Condition", "Block"]] = ("Task", "Task", "Task")
    invalid = values["Time"].isna()
    if invalid.any():
        labels = ", ".join(sorted(values.loc[invalid, "segment"].astype(str).unique()))
        raise ValueError(f"SCHEMA_ERROR: unsupported segment labels: {labels}")
    return values


def eligibility_reason(row: pd.Series) -> str:
    if not np.isfinite(pd.to_numeric(row["nni_corrected_ms"], errors="coerce")):
        return "raw_missing"
    if str(row.get("review_status", "")).casefold() == "excluded":
        return "manual_excluded"
    if str(row.get("qc_status", "")).casefold() in {"excluded", "failed"}:
        return "qc_excluded"
    if str(row.get("correction_action", "")).casefold() == "excluded":
        return "artifact_excluded"
    if str(row.get("correction_action", "")).casefold() == "interpolated":
        return "interpolated"
    if str(row.get("correction_action", "")).casefold() not in {"none", "nan"}:
        return "corrected"
    return "valid"


def validated_nni(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nni = pd.concat(frames, ignore_index=True)
    qc = pd.concat([load_qc_status(Path(path)) for path in nni["input_path"].unique()], ignore_index=True)
    nni = nni.merge(qc, how="left", on=["source_file", "participant", "segment"], validate="many_to_one")
    nni = add_hierarchy(nni)
    nni["nni_corrected_ms"] = pd.to_numeric(nni["nni_corrected_ms"], errors="coerce")
    nni["analysis_reason"] = nni.apply(eligibility_reason, axis=1)
    nni["analysis_eligible"] = nni["analysis_reason"].isin({"valid", "corrected", "interpolated"})
    return nni


def metric_row(group: pd.DataFrame, ddof: int) -> pd.Series:
    values = group.loc[group["analysis_eligible"], "nni_corrected_ms"].to_numpy(dtype=float)
    n = len(values)
    rmssd = np.sqrt(np.mean(np.diff(values) ** 2)) if n >= 2 else np.nan
    sdnn = np.std(values, ddof=ddof) if n > ddof else np.nan
    return pd.Series({
        "n_nni_total": len(group), "n_nni_eligible": n,
        "Mean_RR_ms": np.mean(values) if n else np.nan,
        "SDNN_ms": sdnn, "RMSSD_ms": rmssd,
        "metric_status": "valid" if n >= max(2, ddof + 1) else "insufficient_eligible_nni",
    })


def block_metrics(nni: pd.DataFrame, ddof: int) -> pd.DataFrame:
    keys = ["participant", "Time", "Condition", "Block"]
    return nni.groupby(keys, dropna=False, sort=True).apply(metric_row, ddof=ddof, include_groups=False).reset_index().rename(columns={"participant": "Participant"})


def condition_metrics(blocks: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (participant, time, condition), group in blocks.groupby(["Participant", "Time", "Condition"], sort=True):
        expected = 1 if condition == "Task" else 3
        usable = group.loc[group["metric_status"].eq("valid")]
        complete = len(usable) == expected
        include = complete or policy == "available_blocks"
        row: dict[str, object] = {
            "Participant": participant, "Time": time, "Condition": condition,
            "expected_blocks": expected, "observed_blocks": len(group), "usable_blocks": len(usable),
            "condition_status": "valid" if include and len(usable) else "missing_required_blocks",
        }
        for metric in ("Mean_RR_ms", "SDNN_ms", "RMSSD_ms"):
            row[metric] = usable[metric].mean() if include and len(usable) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def group_summary(conditions: pd.DataFrame, ci_method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (time, condition), group in conditions.loc[conditions["condition_status"].eq("valid")].groupby(["Time", "Condition"], sort=True):
        for metric in ("Mean_RR_ms", "SDNN_ms", "RMSSD_ms"):
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            if n >= 2:
                se = np.std(values, ddof=1) / np.sqrt(n)
                ci_half = stats.t.ppf(0.975, n - 1) * se
            else:
                ci_half = np.nan
            rows.append({"Time": time, "Condition": condition, "metric": metric, "n_participants": n,
                         "mean": mean, "ci_lower_95": mean - ci_half if n >= 2 else np.nan,
                         "ci_upper_95": mean + ci_half if n >= 2 else np.nan, "ci_method": ci_method})
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    paths = discover_inputs(args.input_dir)
    nni = validated_nni([load_nni(path) for path in paths])
    blocks = block_metrics(nni, ddof=int(args.sdnn_ddof))
    conditions = condition_metrics(blocks, args.missing_block_policy)
    summary = group_summary(conditions, args.ci_method)
    exclusions = nni.loc[~nni["analysis_eligible"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nni.to_csv(args.output_dir / "validated_nni.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(args.output_dir / "block_metrics.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(args.output_dir / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(args.output_dir / "analysis_exclusions.csv", index=False, encoding="utf-8-sig")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
                "input_files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
                "decisions": {"sdnn_ddof": int(args.sdnn_ddof), "ci_method": args.ci_method,
                              "missing_block_policy": args.missing_block_policy},
                "python": sys.version, "platform": platform.platform(), "figures_generated": False}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote analysis outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
