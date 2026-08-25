"""Batch ECG/HRV preprocessing for BrainVision recordings.

The script searches an input folder recursively for ``.vhdr`` files, detects the
ECG channel, uses BrainVision marker comments to create pre-rest/task/post-rest
segments, and writes audit-friendly preprocessing outputs.  The original RR
intervals are never overwritten: artifact flags and the corrected NNI are kept
as separate columns.

Install once (if needed):
    pip install mne neurokit2 numpy scipy pandas matplotlib

First copy and complete ``researcher_approved_config.template.json``. Then run:
    python hrv_preprocess.py --config researcher_approved_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permits batch runs on computers without a GUI
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal


PLOT_SECONDS = 20.0


@dataclass(frozen=True)
class Segment:
    """One analysis period expressed in samples in the original recording."""

    name: str
    condition: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path.cwd(), help="Folder containing .vhdr files (default: current folder).")
    parser.add_argument("--output-dir", type=Path, default=Path("hrv_output"), help="Where results are written.")
    parser.add_argument("--ecg-channel", default="ECG", help="Exact ECG channel name. Default: ECG; case-insensitive matching is used.")
    parser.add_argument("--config", type=Path, required=True, help="Researcher-approved JSON configuration. The pipeline will not use implicit methodology defaults.")
    return parser.parse_args()


REQUIRED_CONFIG = {
    "method_config_version": str,
    "filter": {"low_hz": (int, float), "high_hz": (int, float), "order": int, "type": str, "zero_phase": bool},
    "rpeak": {"primary_method": str, "secondary_method": (str, type(None)), "require_cross_check": bool, "match_tolerance_ms": (int, float)},
    "artifact": {"min_rri_ms": (int, float), "max_rri_ms": (int, float), "local_change_threshold": (int, float), "mad_threshold": (int, float)},
    "correction": {"method": str, "max_consecutive_artifacts": int},
    "segment_qc": {"minimum_duration_s": (int, float), "maximum_artifact_percent": (int, float)},
    "rest_state_sequence": list,
}


def load_researcher_config(path: Path) -> tuple[dict, str]:
    """Load only a complete, researcher-approved method configuration."""
    try:
        text = path.read_text(encoding="utf-8")
        config = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"CONFIG_ERROR: cannot read JSON configuration: {exc}") from exc

    def validate(value: object, schema: object, location: str) -> None:
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                raise ValueError(f"CONFIG_ERROR: {location} must be an object.")
            for key, child_schema in schema.items():
                if key not in value or value[key] == "REQUIRES_RESEARCHER_DECISION":
                    raise ValueError(f"REQUIRES RESEARCHER DECISION: configuration value '{location}.{key}' is missing.")
                validate(value[key], child_schema, f"{location}.{key}")
        elif not isinstance(value, schema):
            raise ValueError(f"CONFIG_ERROR: {location} has invalid type.")

    validate(config, REQUIRED_CONFIG, "config")
    if not config["rest_state_sequence"] or any(not isinstance(item, str) or item == "REQUIRES_RESEARCHER_DECISION" for item in config["rest_state_sequence"]):
        raise ValueError("REQUIRES RESEARCHER DECISION: rest_state_sequence must contain approved labels.")
    if config["filter"]["type"] != "bandpass" or not config["filter"]["zero_phase"]:
        raise ValueError("CONFIG_ERROR: this implementation supports only zero-phase bandpass filtering.")
    if config["correction"]["method"] != "linear_interpolation":
        raise ValueError("CONFIG_ERROR: this implementation supports only approved method 'linear_interpolation'.")
    if config["rpeak"]["require_cross_check"] and not config["rpeak"]["secondary_method"]:
        raise ValueError("CONFIG_ERROR: require_cross_check=true requires a secondary_method.")
    if not 0 < config["filter"]["low_hz"] < config["filter"]["high_hz"] or config["filter"]["order"] < 1:
        raise ValueError("CONFIG_ERROR: invalid filter limits or order.")
    if config["artifact"]["min_rri_ms"] >= config["artifact"]["max_rri_ms"]:
        raise ValueError("CONFIG_ERROR: artifact min_rri_ms must be below max_rri_ms.")
    return config, hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_ecg_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    lookup = {name.casefold(): name for name in raw.ch_names}
    if requested.casefold() in lookup:
        return lookup[requested.casefold()]
    candidates = [name for name in raw.ch_names if "ecg" in name.casefold() or "ekg" in name.casefold()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot identify one ECG channel. Available channels: {raw.ch_names}")


def parse_vmrk(vmrk_path: Path) -> pd.DataFrame:
    """Read BrainVision markers without depending on event-code conversion."""
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"^Mk\d+=(.*?),(.*?),(\d+),(\d+),(\d+)(?:,.*)?$")
    with vmrk_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                marker_type, description, position, size, channel = match.groups()
                rows.append(
                    {
                        "type": marker_type,
                        "description": description,
                        "sample": int(position) - 1,  # BrainVision marker positions are one-based.
                        "size": int(size),
                        "channel": int(channel),
                    }
                )
    return pd.DataFrame(rows, columns=["type", "description", "sample", "size", "channel"])


def marker_segments(markers: pd.DataFrame, n_samples: int, rest_states: list[str]) -> list[Segment]:
    """Create rests before/after task plus the complete task period.

    Rest markers do not encode EO/EC. Labels therefore come exclusively from the
    researcher-approved ``rest_state_sequence``. A count mismatch excludes the
    entire file rather than cycling or guessing labels.
    """
    marker_text = markers["description"].astype(str).str.casefold()
    task_start = markers.loc[marker_text.eq("task start"), "sample"].tolist()
    task_end = markers.loc[marker_text.eq("task end"), "sample"].tolist()
    if len(task_start) != 1 or len(task_end) != 1 or task_end[0] <= task_start[0]:
        raise ValueError("Expected exactly one valid 'task start' and 'task end' marker.")

    starts = markers.loc[marker_text.eq("resting start"), "sample"].tolist()
    ends = markers.loc[marker_text.eq("resting end"), "sample"].tolist()
    if len(starts) != len(ends):
        raise ValueError(f"Unpaired resting markers: {len(starts)} starts and {len(ends)} ends.")

    rests: list[tuple[int, int]] = []
    end_index = 0
    for start in starts:
        while end_index < len(ends) and ends[end_index] <= start:
            end_index += 1
        if end_index == len(ends):
            raise ValueError("A 'resting start' has no following 'resting end'.")
        rests.append((start, ends[end_index]))
        end_index += 1

    pre_rests = [(start, end) for start, end in rests if end <= task_start[0]]
    post_rests = [(start, end) for start, end in rests if start >= task_end[0]]
    overlapping = len(rests) - len(pre_rests) - len(post_rests)
    if overlapping:
        raise ValueError("MARKER_ERROR: resting period overlaps the task window.")
    if len(pre_rests) != len(rest_states) or len(post_rests) != len(rest_states):
        raise ValueError(
            f"MARKER_ERROR: expected {len(rest_states)} pre-task and post-task resting segments; "
            f"found {len(pre_rests)} and {len(post_rests)}. File excluded; labels were not inferred."
        )

    segments: list[Segment] = [Segment("task", "task", task_start[0], task_end[0])]
    for pre_i, (start, end) in enumerate(pre_rests, start=1):
        start, end = max(0, start), min(n_samples, end)
        if end <= start:
            raise ValueError("SEGMENT_ERROR: an approved pre-task resting segment is empty.")
        state = rest_states[pre_i - 1]
        segments.append(Segment(f"pre_rest_{pre_i:02d}_{state}", f"pre_rest_{state}", start, end))
    for post_i, (start, end) in enumerate(post_rests, start=1):
        start, end = max(0, start), min(n_samples, end)
        if end <= start:
            raise ValueError("SEGMENT_ERROR: an approved post-task resting segment is empty.")
        state = rest_states[post_i - 1]
        segments.append(Segment(f"post_rest_{post_i:02d}_{state}", f"post_rest_{state}", start, end))
    return segments


def bandpass_ecg(ecg: np.ndarray, sampling_rate: float, config: dict) -> np.ndarray:
    filter_config = config["filter"]
    low, high = filter_config["low_hz"], filter_config["high_hz"]
    if high >= sampling_rate / 2 or low <= 0:
        raise ValueError(f"FILTER_ERROR: approved cutoff [{low}, {high}] Hz is invalid for {sampling_rate} Hz data.")
    sos = signal.butter(filter_config["order"], [low, high], btype=filter_config["type"], fs=sampling_rate, output="sos")
    return signal.sosfiltfilt(sos, ecg)


def detect_artifacts(rri_ms: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return artifact mask and a semicolon-separated reason for each RRI.

    Criteria combine physiological plausibility, adjacent beat changes, and a
    robust global (MAD) outlier rule.  A beat can be flagged by more than one.
    """
    n = len(rri_ms)
    settings = config["artifact"]
    physiological = ~np.isfinite(rri_ms) | (rri_ms < settings["min_rri_ms"]) | (rri_ms > settings["max_rri_ms"])
    global_outlier = np.zeros(n, dtype=bool)
    local_change = np.zeros(n, dtype=bool)
    if n < 3:
        flags = {"artifact_physiological_range": physiological, "artifact_global_mad": global_outlier, "artifact_local_change": local_change}
        return physiological, flags, np.where(physiological, "physiological_range", "none")

    median = np.nanmedian(rri_ms)
    mad = np.nanmedian(np.abs(rri_ms - median))
    if np.isfinite(mad) and mad > 0:
        robust_z = 0.6745 * (rri_ms - median) / mad
        global_outlier = np.abs(robust_z) > settings["mad_threshold"]

    neighbour_median = (rri_ms[:-2] + rri_ms[2:]) / 2
    local_change[1:-1] = np.abs(rri_ms[1:-1] - neighbour_median) > settings["local_change_threshold"] * neighbour_median
    invalid = physiological | global_outlier | local_change
    flags = {"artifact_physiological_range": physiological, "artifact_global_mad": global_outlier, "artifact_local_change": local_change}
    reasons = np.array([";".join(key.removeprefix("artifact_") for key, values in flags.items() if values[index]) or "none" for index in range(n)], dtype=object)
    return invalid, flags, reasons


def correct_artifacts(rri_ms: np.ndarray, invalid: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correct only runs allowed by the approved configuration; retain all raw RRI."""
    corrected = rri_ms.astype(float).copy()
    corrected[invalid] = np.nan
    interpolated = np.zeros(len(rri_ms), dtype=bool)
    action = np.full(len(rri_ms), "none", dtype=object)
    method = np.full(len(rri_ms), "none", dtype=object)
    action[invalid] = "excluded"
    maximum = config["correction"]["max_consecutive_artifacts"]
    index = 0
    while index < len(rri_ms):
        if not invalid[index]:
            index += 1
            continue
        end = index
        while end < len(rri_ms) and invalid[end]:
            end += 1
        run_length = end - index
        if index > 0 and end < len(rri_ms) and run_length <= maximum and not invalid[index - 1] and not invalid[end]:
            corrected[index:end] = np.linspace(rri_ms[index - 1], rri_ms[end], run_length + 2)[1:-1]
            interpolated[index:end] = True
            action[index:end] = "interpolated"
            method[index:end] = config["correction"]["method"]
        index = end
    return corrected, interpolated, action, method


def qc_metrics(raw_ecg: np.ndarray, filtered_ecg: np.ndarray, sampling_rate: float) -> dict[str, float]:
    duration_s = len(raw_ecg) / sampling_rate
    slope_per_second = np.polyfit(np.arange(len(raw_ecg)) / sampling_rate, raw_ecg, 1)[0] if len(raw_ecg) > 1 else np.nan
    full_scale_fraction = float(np.mean(np.abs(raw_ecg) >= 0.999 * np.nanmax(np.abs(raw_ecg)))) if np.nanmax(np.abs(raw_ecg)) else 0.0
    dropout_fraction = float(np.mean(np.abs(np.diff(raw_ecg)) < np.finfo(float).eps)) if len(raw_ecg) > 1 else np.nan
    return {
        "duration_s": duration_s,
        "raw_sd": float(np.std(raw_ecg)),
        "filtered_sd": float(np.std(filtered_ecg)),
        "baseline_slope_per_s": float(slope_per_second),
        "near_segment_amplitude_ceiling_fraction": full_scale_fraction,
        "consecutive_identical_sample_fraction": dropout_fraction,
    }


def save_ecg_plot(path: Path, raw_ecg: np.ndarray, filtered_ecg: np.ndarray, rpeaks: np.ndarray, sampling_rate: float, title: str) -> None:
    count = min(len(raw_ecg), int(PLOT_SECONDS * sampling_rate))
    t = np.arange(count) / sampling_rate
    visible = rpeaks[rpeaks < count]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, raw_ecg[:count], lw=0.6, color="0.35")
    axes[0].set_title(f"{title}: raw ECG (first {count / sampling_rate:.0f} s)")
    axes[1].plot(t, filtered_ecg[:count], lw=0.7, color="#1769aa")
    axes[1].scatter(visible / sampling_rate, filtered_ecg[visible], s=20, c="#d32f2f", label="R peaks", zorder=3)
    axes[1].set_title("0.5–40 Hz filtered ECG with detected R peaks")
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_rri_plot(path: Path, rri: pd.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    axes[0].plot(rri["time_s"], rri["rri_ms"], ".-", ms=3, lw=0.7, label="Original RRI")
    flagged = rri["artifact"]
    axes[0].scatter(rri.loc[flagged, "time_s"], rri.loc[flagged, "rri_ms"], c="#d32f2f", s=22, label="Artifact", zorder=3)
    axes[0].legend()
    axes[0].set_title(f"{title}: original RRI and detected artifacts")
    axes[1].plot(rri["time_s"], rri["corrected_nni_ms"], ".-", ms=3, lw=0.7, color="#1769aa", label="Corrected NNI")
    axes[1].scatter(rri.loc[rri["interpolated"], "time_s"], rri.loc[rri["interpolated"], "corrected_nni_ms"], c="#f57c00", s=22, label="Isolated interpolation", zorder=3)
    axes[1].legend()
    axes[1].set_title("Corrected NNI (unresolved artifacts remain missing)")
    axes[1].set_xlabel("Time from segment start (s)")
    for axis in axes:
        axis.set_ylabel("Interval (ms)")
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rpeak_agreement(primary: np.ndarray, secondary: np.ndarray, tolerance_samples: int) -> np.ndarray:
    """Return whether each primary peak has a secondary peak within tolerance."""
    if not len(secondary):
        return np.zeros(len(primary), dtype=bool)
    positions = np.searchsorted(secondary, primary)
    agreement = np.zeros(len(primary), dtype=bool)
    for index, position in enumerate(positions):
        candidates = secondary[max(0, position - 1) : min(len(secondary), position + 1)]
        agreement[index] = bool(len(candidates) and np.min(np.abs(candidates - primary[index])) <= tolerance_samples)
    return agreement


def process_segment(ecg: np.ndarray, sampling_rate: float, segment: Segment, source_name: str, participant: str, output_dir: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_piece = ecg[segment.start : segment.end]
    if len(raw_piece) / sampling_rate < config["segment_qc"]["minimum_duration_s"]:
        raise ValueError("INSUFFICIENT_DATA: segment duration is below approved minimum_duration_s.")
    filtered = bandpass_ecg(raw_piece, sampling_rate, config)
    rpeak_config = config["rpeak"]
    _, info = nk.ecg_peaks(filtered, sampling_rate=sampling_rate, method=rpeak_config["primary_method"], correct_artifacts=False)
    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    secondary = np.array([], dtype=int)
    agreement = np.ones(len(rpeaks), dtype=bool)
    if rpeak_config["require_cross_check"]:
        _, secondary_info = nk.ecg_peaks(filtered, sampling_rate=sampling_rate, method=rpeak_config["secondary_method"], correct_artifacts=False)
        secondary = np.asarray(secondary_info["ECG_R_Peaks"], dtype=int)
        agreement = rpeak_agreement(rpeaks, secondary, round(rpeak_config["match_tolerance_ms"] * sampling_rate / 1000))
    peak_review = np.where(agreement, "not_reviewed", "needs_review")
    raw_peaks = pd.concat([
        pd.DataFrame({
            "source_file": source_name, "participant": participant, "segment": segment.name, "condition": segment.condition,
            "rpeak_index": np.arange(len(rpeaks)), "rpeak_sample": rpeaks + segment.start,
            "rpeak_time_s": (rpeaks + segment.start) / sampling_rate, "rpeak_detector": rpeak_config["primary_method"],
            "rpeak_primary_secondary_agreement": agreement, "rpeak_review_status": peak_review,
            "reviewer": "", "review_timestamp": "", "review_decision": "", "review_note": "",
        }),
        pd.DataFrame({
            "source_file": source_name, "participant": participant, "segment": segment.name, "condition": segment.condition,
            "rpeak_index": np.arange(len(secondary)), "rpeak_sample": secondary + segment.start,
            "rpeak_time_s": (secondary + segment.start) / sampling_rate, "rpeak_detector": rpeak_config["secondary_method"],
            "rpeak_primary_secondary_agreement": [bool(np.any(np.abs(rpeaks - item) <= round(rpeak_config["match_tolerance_ms"] * sampling_rate / 1000))) for item in secondary],
            "rpeak_review_status": "not_reviewed", "reviewer": "", "review_timestamp": "", "review_decision": "", "review_note": "",
        }) if len(secondary) else pd.DataFrame(),
    ], ignore_index=True)
    rri_ms = np.diff(rpeaks) / sampling_rate * 1000
    invalid, flags, reasons = detect_artifacts(rri_ms, config)
    corrected, interpolated, action, correction_method = correct_artifacts(rri_ms, invalid, config)
    artifact_type = np.where(~invalid, "none", np.where(np.sum(np.column_stack(list(flags.values())), axis=1) == 1, reasons, "uncertain"))
    interval_review = np.where(agreement[:-1] & agreement[1:], "not_reviewed", "needs_review") if len(rri_ms) else np.array([], dtype=object)
    rri = pd.DataFrame(
        {
            "source_file": source_name,
            "participant": participant,
            "segment": segment.name,
            "condition": segment.condition,
            "previous_rpeak_sample": rpeaks[:-1] + segment.start,
            "previous_rpeak_time_s": (rpeaks[:-1] + segment.start) / sampling_rate,
            "next_rpeak_sample": rpeaks[1:] + segment.start,
            "next_rpeak_time_s": (rpeaks[1:] + segment.start) / sampling_rate,
            "rri_raw_ms": rri_ms,
            "artifact": invalid,
            "artifact_type": artifact_type,
            "artifact_reason": reasons,
            **flags,
            "correction_action": action,
            "correction_method": correction_method,
            "interpolated": interpolated,
            "nni_corrected_ms": corrected,
            "correction_source_index": np.where(interpolated, np.arange(len(rri_ms)), np.nan),
            "correction_review_status": np.where(interpolated, "needs_review", "not_reviewed"),
            "review_status": interval_review,
            "reviewer": "", "review_timestamp": "", "review_decision": "", "review_note": "",
        }
    )
    qc = qc_metrics(raw_piece, filtered, sampling_rate)
    metrics: dict[str, object] = {
        "source_file": source_name, "participant": participant,
        "segment": segment.name,
        "condition": segment.condition,
        "n_rpeaks": len(rpeaks),
        "n_intervals": len(rri_ms),
        "n_artifacts": int(invalid.sum()),
        "n_interpolated": int(interpolated.sum()),
        "n_unresolved": int((invalid & ~interpolated).sum()),
        "artifact_percent": float(invalid.mean() * 100) if len(invalid) else np.nan,
        "quality_score": float(100 - invalid.mean() * 100) if len(invalid) else np.nan,
        "rpeak_method": rpeak_config["primary_method"], "rpeak_secondary_method": rpeak_config["secondary_method"],
        "rpeak_disagreement_count": int((~agreement).sum()),
        "filter_low_hz": config["filter"]["low_hz"], "filter_high_hz": config["filter"]["high_hz"],
        "filter_order": config["filter"]["order"], "filter_type": config["filter"]["type"], "zero_phase": config["filter"]["zero_phase"],
        "sampling_rate_hz": sampling_rate,
        "status": "needs_review" if ((~agreement).any() or invalid.mean() * 100 > config["segment_qc"]["maximum_artifact_percent"]) else "not_reviewed",
        "exclusion_reason": "artifact_percent_exceeds_approved_limit" if len(invalid) and invalid.mean() * 100 > config["segment_qc"]["maximum_artifact_percent"] else "",
        **qc,
    }
    save_ecg_plot(output_dir / f"{segment.name}_ecg_qc.png", raw_piece, filtered, rpeaks, sampling_rate, source_name)
    # Plot uses relative segment time while the CSV holds unambiguous absolute R-peak times.
    plot_rri = rri.assign(time_s=rpeaks[1:] / sampling_rate, rri_ms=rri["rri_raw_ms"], corrected_nni_ms=rri["nni_corrected_ms"])
    save_rri_plot(output_dir / f"{segment.name}_rri_qc.png", plot_rri, source_name)
    return rri, raw_peaks, metrics


def process_file(vhdr_path: Path, output_root: Path, requested_channel: str, config: dict, config_hash: str) -> pd.DataFrame:
    subject_dir = output_root / vhdr_path.stem
    subject_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    ecg_channel = find_ecg_channel(raw, requested_channel)
    sampling_rate = float(raw.info["sfreq"])
    ecg = raw.get_data(picks=[ecg_channel])[0]
    markers = parse_vmrk(vhdr_path.with_suffix(".vmrk"))
    markers.to_csv(subject_dir / "markers.csv", index=False, encoding="utf-8-sig")
    segments = marker_segments(markers, len(ecg), config["rest_state_sequence"])
    pd.DataFrame([segment.__dict__ for segment in segments]).assign(sampling_rate_hz=sampling_rate, ecg_channel=ecg_channel).to_csv(subject_dir / "segments.csv", index=False, encoding="utf-8-sig")

    all_rri: list[pd.DataFrame] = []
    all_peaks: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    for segment in segments:
        try:
            rri, peaks, metrics = process_segment(ecg, sampling_rate, segment, vhdr_path.name, vhdr_path.stem, subject_dir, config)
            all_rri.append(rri)
            all_peaks.append(peaks)
            metrics.update({"method_config_version": config["method_config_version"], "config_sha256": config_hash})
            all_metrics.append(metrics)
        except Exception as exc:  # continue so one bad segment does not hide other usable periods
            all_metrics.append({"source_file": vhdr_path.name, "participant": vhdr_path.stem, "segment": segment.name, "condition": segment.condition, "status": "excluded", "exclusion_reason": str(exc), "error": str(exc)})
            warnings.warn(f"{vhdr_path.name}, {segment.name}: {exc}")
    rri_frame = pd.concat(all_rri, ignore_index=True) if all_rri else pd.DataFrame()
    peaks_frame = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()
    rri_frame.to_csv(subject_dir / "rri_and_nni.csv", index=False, encoding="utf-8-sig")
    peaks_frame.to_csv(subject_dir / "raw_rpeaks.csv", index=False, encoding="utf-8-sig")
    qc = pd.DataFrame(all_metrics)
    qc.to_csv(subject_dir / "preprocessing_qc.csv", index=False, encoding="utf-8-sig")
    versions = {"python_version": sys.version, "mne_version": mne.__version__, "neurokit2_version": nk.__version__, "numpy_version": np.__version__, "scipy_version": __import__("scipy").__version__, "pandas_version": pd.__version__, "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(), "method_config_version": config["method_config_version"], "config_sha256": config_hash}
    (subject_dir / "reproducibility.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    (subject_dir / "processing_log.txt").write_text("\n".join(f"{row.get('segment', 'file')}: {row.get('status', 'processed')} {row.get('exclusion_reason', '')}" for row in all_metrics), encoding="utf-8")
    return qc


def main() -> int:
    args = parse_args()
    try:
        config, config_hash = load_researcher_config(args.config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else input_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vhdr_files = sorted(path for path in input_dir.rglob("*.vhdr") if output_dir not in path.parents)
    if not vhdr_files:
        print(f"No .vhdr files found under {input_dir}", file=sys.stderr)
        return 2
    all_results: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for vhdr_path in vhdr_files:
        try:
            all_results.append(process_file(vhdr_path, output_dir, args.ecg_channel, config, config_hash))
        except Exception as exc:
            failures.append({"source_file": str(vhdr_path), "error": str(exc)})
            warnings.warn(f"Skipping {vhdr_path.name}: {exc}")
    summary = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary.to_csv(output_dir / "preprocessing_qc_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures, columns=["source_file", "error"]).to_csv(output_dir / "processing_failures.csv", index=False, encoding="utf-8-sig")
    print(f"Done. Results: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
