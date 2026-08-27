"""BrainVision EEG event-related analysis using Stimulus (S) markers only.

This script is tailored to the recording in this workspace.  It treats each
``Stimulus,S <code>`` marker as an independent trial, ignores every ``R``
marker, computes baseline-corrected ERP averages, and summarizes canonical EEG
band power from the same epochs.

Install once:
	python -m pip install mne numpy pandas scipy matplotlib

Run from this folder:
	python try.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


# === 1. 分析參數：定義資料位置、epoch 時窗與 EEG 頻帶 =========================
# S marker 是刺激出現時間；-200 到 0 ms 作為 baseline，0 到 800 ms 用於 ERP。
DEFAULT_INPUT = Path(__file__).resolve().with_name("0727 chen_try.vhdr")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("eeg_analysis_output")
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
BASELINE = (None, 0.0)
BANDS = {
	"delta_1_4Hz": (1.0, 4.0),
	"theta_4_8Hz": (4.0, 8.0),
	"alpha_8_13Hz": (8.0, 13.0),
	"beta_13_30Hz": (13.0, 30.0),
}


def parse_args() -> argparse.Namespace:
	"""=== 2. 命令列介面：允許替換輸入檔與輸出資料夾 =============================="""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="BrainVision .vhdr file")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder")
	return parser.parse_args()


def read_s_markers(vmrk_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
	"""=== 3. Marker audit：只保留 Stimulus/S，明確排除 Response/R ==================

	BrainVision 的位置以 1 起算，因此轉成 0-based sample。描述文字中的
	空白會被正規化，避免 ``S  5`` 和 ``S 5`` 被視為不同事件。
	"""
	rows: list[dict[str, object]] = []
	line_pattern = re.compile(r"^Mk\d+=(.*?),(.*?),(\d+),(\d+),(\d+)(?:,.*)?$")
	with vmrk_path.open("r", encoding="utf-8", errors="replace") as handle:
		for line in handle:
			match = line_pattern.match(line.strip())
			if match:
				marker_type, description, position, size, channel = match.groups()
				normalized = re.sub(r"\s+", " ", description.strip())
				rows.append({
					"type": marker_type,
					"description": description,
					"normalized_description": normalized,
					"sample": int(position) - 1,
					"size": int(size),
					"channel": int(channel),
					"used_for_analysis": bool(marker_type.casefold() == "stimulus" and normalized.casefold().startswith("s ")),
				})

	markers = pd.DataFrame(rows)
	if markers.empty:
		raise ValueError(f"No readable markers found in {vmrk_path}")
	s_markers = markers.loc[markers["used_for_analysis"]].copy()
	s_markers["event_code"] = s_markers["normalized_description"].str.extract(r"(?i)^s\s+(\d+)$", expand=False)
	s_markers = s_markers.loc[s_markers["event_code"].notna()].copy()
	if s_markers.empty:
		raise ValueError("No Stimulus/S markers were found; analysis cannot proceed.")
	s_markers["event_code"] = s_markers["event_code"].astype(int)
	counts = s_markers["event_code"].value_counts().sort_index().to_dict()
	markers["event_code"] = pd.NA
	markers.loc[s_markers.index, "event_code"] = s_markers["event_code"]
	return markers, {str(code): int(count) for code, count in counts.items()}


def make_events(markers: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
	"""=== 4. 事件矩陣：把 S code 轉成 MNE events，R 不可能進入此矩陣 =============="""
	selected = markers.loc[markers["used_for_analysis"] & markers["event_code"].notna()].copy()
	selected = selected.sort_values("sample")
	codes = sorted(selected["event_code"].astype(int).unique())
	event_id = {f"S_{code}": 100 + code for code in codes}
	events = np.column_stack([
		selected["sample"].to_numpy(dtype=int),
		np.zeros(len(selected), dtype=int),
		selected["event_code"].astype(int).map(lambda code: event_id[f"S_{code}"]).to_numpy(dtype=int),
	])
	return events, event_id


def load_and_epoch(vhdr_path: Path, events: np.ndarray, event_id: dict[str, int]) -> tuple[mne.io.BaseRaw, mne.Epochs]:
	"""=== 5. 前處理與切段：濾波、平均參考、baseline correction、壞段排除 ============"""
	raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
	eeg_channels = [name for name in raw.ch_names if raw.get_channel_types(picks=[name])[0] == "eeg"]
	if not eeg_channels:
		raise ValueError("No EEG channels were found in the BrainVision file.")
	raw.pick(eeg_channels)
	raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
	raw.filter(l_freq=1.0, h_freq=40.0, method="fir", phase="zero", verbose="ERROR")
	raw.set_eeg_reference("average", projection=False, verbose="ERROR")
	epochs = mne.Epochs(
		raw,
		events,
		event_id=event_id,
		tmin=EPOCH_TMIN,
		tmax=EPOCH_TMAX,
		baseline=BASELINE,
		reject=dict(eeg=150e-6),
		preload=True,
		reject_by_annotation=True,
		detrend=1,
		verbose="ERROR",
	)
	return raw, epochs


def compute_erp(epochs: mne.Epochs) -> pd.DataFrame:
	"""=== 6. ERP：依 S code 與通道計算平均波形、峰值和峰值延遲 ======================="""
	rows: list[dict[str, object]] = []
	times_ms = epochs.times * 1000.0
	for condition in epochs.event_id:
		condition_epochs = epochs[condition]
		if len(condition_epochs) == 0:
			continue
		evoked = condition_epochs.average()
		analysis_window = (times_ms >= 0) & (times_ms <= 800)
		for channel_index, channel_name in enumerate(evoked.ch_names):
			waveform = evoked.data[channel_index] * 1e6
			window_values = waveform[analysis_window]
			window_times = times_ms[analysis_window]
			peak_index = int(np.argmax(np.abs(window_values)))
			rows.append({
				"event": condition,
				"channel": channel_name,
				"n_epochs": len(condition_epochs),
				"peak_amplitude_uv": float(window_values[peak_index]),
				"peak_latency_ms": float(window_times[peak_index]),
				"mean_amplitude_0_800ms_uv": float(np.mean(window_values)),
			})
	return pd.DataFrame(rows)


def compute_band_power(epochs: mne.Epochs) -> pd.DataFrame:
	"""=== 7. 頻域摘要：用 Welch PSD 計算每個 S code/通道的相對頻帶功率 =============="""
	rows: list[dict[str, object]] = []
	sfreq = epochs.info["sfreq"]
	data = epochs.get_data(copy=True)
	times = epochs.times
	analysis_mask = (times >= 0) & (times <= 0.8)
	for condition, event_code in epochs.event_id.items():
		epoch_indices = np.flatnonzero(epochs.events[:, 2] == event_code)
		if len(epoch_indices) == 0:
			continue
		for channel_index, channel_name in enumerate(epochs.ch_names):
			frequencies, psd = welch(data[epoch_indices, channel_index][:, analysis_mask], fs=sfreq, axis=-1, nperseg=min(512, analysis_mask.sum()))
			total_mask = (frequencies >= 1) & (frequencies <= 40)
			total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=-1)
			for band, (low, high) in BANDS.items():
				band_mask = (frequencies >= low) & (frequencies < high)
				band_power = np.trapezoid(psd[:, band_mask], frequencies[band_mask], axis=-1)
				rows.append({
					"event": condition,
					"channel": channel_name,
					"band": band,
					"n_epochs": len(epoch_indices),
					"absolute_power_uv2_per_hz": float(np.mean(band_power) * 1e12),
					"relative_power": float(np.mean(band_power / total_power)),
				})
	return pd.DataFrame(rows)


def save_figures(epochs: mne.Epochs, output_dir: Path) -> None:
	"""=== 8. 圖形：輸出整體 ERP 與各 S code 的 trial 數，方便品質檢查 ================"""
	evoked = epochs.average()
	figure = evoked.plot(spatial_colors=True, show=False, time_unit="s", titles="S-marker grand-average ERP")
	figure.savefig(output_dir / "erp_grand_average.png", dpi=160, bbox_inches="tight")
	plt.close(figure)
	counts = pd.Series(epochs.events[:, 2]).map({value: key for key, value in epochs.event_id.items()}).value_counts().sort_index()
	axis = counts.plot.bar(figsize=(10, 4), color="#1769aa")
	axis.set_xlabel("Stimulus marker")
	axis.set_ylabel("Accepted epochs")
	axis.set_title("Epoch retention by S marker (R markers excluded)")
	axis.figure.tight_layout()
	axis.figure.savefig(output_dir / "epoch_retention.png", dpi=160)
	plt.close(axis.figure)


def main() -> int:
	"""=== 9. 主流程：建立 audit、分析表格與可重現輸出 ================================"""
	args = parse_args()
	output_dir = args.output.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	markers, marker_counts = read_s_markers(args.input.with_suffix(".vmrk"))
	events, event_id = make_events(markers)
	_, epochs = load_and_epoch(args.input, events, event_id)
	markers.to_csv(output_dir / "marker_audit.csv", index=False, encoding="utf-8-sig")
	compute_erp(epochs).to_csv(output_dir / "erp_metrics.csv", index=False, encoding="utf-8-sig")
	compute_band_power(epochs).to_csv(output_dir / "band_power.csv", index=False, encoding="utf-8-sig")
	save_figures(epochs, output_dir)
	pd.DataFrame([{
		"input_file": str(args.input),
		"sampling_rate_hz": epochs.info["sfreq"],
		"eeg_channels": len(epochs.ch_names),
		"s_markers_used": len(events),
		"r_markers_ignored": int((markers["type"].astype(str).str.casefold() == "response").sum()),
		"accepted_epochs": len(epochs),
		"event_counts": str(marker_counts),
		"filter_hz": "1-40",
		"baseline_s": "-0.2-0.0",
		"epoch_s": "-0.2-0.8",
	}]).to_csv(output_dir / "analysis_manifest.csv", index=False, encoding="utf-8-sig")
	print(f"完成。S markers: {len(events)}；accepted epochs: {len(epochs)}；輸出: {output_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
