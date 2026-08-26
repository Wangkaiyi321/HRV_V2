"""HRV time-domain analysis from preprocessed RRI/NNI files.

This program deliberately does *not* filter ECG, detect R peaks, flag
artifacts, or correct intervals.  Those operations belong to
``hrv_preprocess.py``.  It reads each ``rri_and_nni.csv`` below an input
folder and produces auditable NNI, block-level, condition-level, and group
summary tables.

Example (the three arguments below are researcher decisions)::

    python hrv_time_domain.py --input-dir hrv_output --sdnn-ddof 1 \
        --ci-method t --missing-block-policy complete_blocks

No significance tests, p values, multiple-comparison corrections, or effect
sizes are implemented here.  Figures show descriptive means and 95% CIs only.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_NNI_COLUMNS = {
    "source_file", "participant", "segment", "condition",
    "nni_corrected_ms", "correction_action", "review_status",
}
REST_SEGMENT = r"^(pre|post)_rest_\d+_(EC|EO)$"


def prompt_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    options = "/".join(choices)
    while True:
        value = input(f"{name} [{options}] (default: {default}): ").strip() or default
        if value in choices:
            return value
        print(f"Please enter one of: {options}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("hrv_preprocess_output"), help="Folder containing rri_and_nni.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path("hrv_time_domain_output"), help="Folder for analysis outputs.")
    parser.add_argument("--sdnn-ddof", default="1", choices=("0", "1"), help="Approved SDNN estimator: 0=population, 1=sample (default: 1).")
    parser.add_argument("--ci-method", default="t", choices=("t",), help="Approved 95%% CI estimator (default: t).")
    parser.add_argument("--missing-block-policy", choices=("complete_blocks", "available_blocks"), help="Researcher-approved handling for missing resting blocks.")
    args = parser.parse_args()
    if args.missing_block_policy is None:
        args.missing_block_policy = prompt_choice(
            "Missing block policy", ("complete_blocks", "available_blocks"), "complete_blocks"
        )
    return args


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
    eligible = group["analysis_eligible"].to_numpy(dtype=bool)
    raw_values = group["nni_corrected_ms"].to_numpy(dtype=float)
    values = raw_values[eligible]
    n = len(values)

    # RMSSD must only use successive-difference pairs where BOTH beats are
    # temporally adjacent (original order) AND eligible. Filtering out
    # ineligible beats first and then taking np.diff on what remains would
    # compute a "difference" across the gap left by the excluded beat(s),
    # which is not a real successive-beat difference and inflates RMSSD.
    if len(raw_values) >= 2:
        pair_eligible = eligible[:-1] & eligible[1:]
        successive_diffs = np.diff(raw_values)[pair_eligible]
    else:
        successive_diffs = np.array([])
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) >= 1 else np.nan

    sdnn = np.std(values, ddof=ddof) if n > ddof else np.nan
    return pd.Series({
        "n_nni_total": len(group), "n_nni_eligible": n,
        "n_successive_pairs_eligible": len(successive_diffs),
        "Mean_RR_ms": np.mean(values) if n else np.nan,
        "SDNN_ms": sdnn,
        "RMSSD_ms": rmssd,
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


def block_group_summary(blocks: pd.DataFrame, ci_method: str) -> pd.DataFrame:
    """Participant-level block metrics aggregated for descriptive plotting."""
    rows: list[dict[str, object]] = []
    keys = ["Time", "Condition", "Block"]
    for key_values, group in blocks.loc[blocks["metric_status"].eq("valid")].groupby(keys, sort=True):
        for metric in ("Mean_RR_ms", "SDNN_ms", "RMSSD_ms"):
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = np.mean(values) if n else np.nan
            ci_half = stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n) if n >= 2 else np.nan
            rows.append({"Time": key_values[0], "Condition": key_values[1], "Block": key_values[2],
                         "metric": metric, "n_participants": n, "mean": mean,
                         "ci_lower_95": mean - ci_half if n >= 2 else np.nan,
                         "ci_upper_95": mean + ci_half if n >= 2 else np.nan, "ci_method": ci_method})
    return pd.DataFrame(rows)


METRIC_LABELS = {"Mean_RR_ms": "Mean RR (ms)", "SDNN_ms": "SDNN (ms)", "RMSSD_ms": "RMSSD (ms)"}
CONDITION_ORDER = [("Pre", "EC"), ("Pre", "EO"), ("Post", "EC"), ("Post", "EO"), ("Task", "Task")]


def draw_line(ax: plt.Axes, frame: pd.DataFrame, labels: list[str]) -> None:
    y = frame["mean"].to_numpy(dtype=float)
    lower = frame["ci_lower_95"].to_numpy(dtype=float)
    upper = frame["ci_upper_95"].to_numpy(dtype=float)
    yerr = np.vstack((y - lower, upper - y))
    yerr[:, ~np.isfinite(yerr).all(axis=0)] = 0
    ax.errorbar(range(len(frame)), y, yerr=yerr, marker="o", linewidth=2, capsize=4, color="#2B6CB0")
    ax.set_xticks(range(len(frame)), labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)


def create_figures(block_summary: pd.DataFrame, condition_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    """Create the six requested descriptive line charts, without inference marks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for metric, stem in (("Mean_RR_ms", "mean_rr"), ("SDNN_ms", "sdnn"), ("RMSSD_ms", "rmssd")):
        condition_source = condition_summary.loc[condition_summary["metric"].eq(metric)].set_index(["Time", "Condition"])
        condition_frame = condition_source.reindex(pd.MultiIndex.from_tuples(CONDITION_ORDER, names=["Time", "Condition"])).reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_line(ax, condition_frame, [f"{time}\n{condition}" for time, condition in CONDITION_ORDER])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} — Condition level")
        fig.tight_layout()
        condition_path = output_dir / f"{stem}_condition_level.png"
        fig.savefig(condition_path, dpi=300)
        plt.close(fig)

        source = block_summary.loc[block_summary["metric"].eq(metric)].copy()
        for time in ("Pre", "Post"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
            frame = source.loc[source["Time"].eq(time)].copy()
            frame["block_number"] = frame["Block"].str.extract(r"(\d+)$").astype(int)
            for ax, condition in zip(axes, ("EC", "EO"), strict=True):
                panel = frame.loc[frame["Condition"].eq(condition)].sort_values("block_number")
                draw_line(ax, panel, panel["Block"].tolist())
                ax.set_title(condition)
            axes[0].set_ylabel(METRIC_LABELS[metric])
            fig.suptitle(f"{METRIC_LABELS[metric]} — {time} block level", y=1.02)
            fig.tight_layout()
            block_path = output_dir / f"{stem}_block_{time.casefold()}.png"
            fig.savefig(block_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            names.append(block_path.name)
        names.append(condition_path.name)
    return names


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    paths = discover_inputs(args.input_dir)
    nni = validated_nni([load_nni(path) for path in paths])
    blocks = block_metrics(nni, ddof=int(args.sdnn_ddof))
    conditions = condition_metrics(blocks, args.missing_block_policy)
    summary = group_summary(conditions, args.ci_method)
    block_summary = block_group_summary(blocks, args.ci_method)
    exclusions = nni.loc[~nni["analysis_eligible"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nni.to_csv(args.output_dir / "validated_nni.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(args.output_dir / "block_metrics.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(args.output_dir / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
    block_summary.to_csv(args.output_dir / "block_group_summary.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(args.output_dir / "analysis_exclusions.csv", index=False, encoding="utf-8-sig")
    figure_names = create_figures(block_summary, summary, args.output_dir / "figures")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
                "input_files": [{"path": str(path), "sha256": sha256(path)} for path in paths],
                "decisions": {"sdnn_ddof": int(args.sdnn_ddof), "ci_method": args.ci_method,
                              "missing_block_policy": args.missing_block_policy},
                "python": sys.version, "platform": platform.platform(), "figures_generated": figure_names}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote analysis outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
