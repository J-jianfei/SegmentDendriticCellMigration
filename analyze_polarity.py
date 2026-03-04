#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


@dataclass
class TrackPolarity:
    source: str
    track_id: int
    class_name: str
    n_steps: int
    n_frames: int
    mean_polarity: float
    abs_mean_polarity: float
    directed_persistence: float
    net_displacement_px: float
    path_length_px: float
    reversals: int
    reversal_rate_per_step: float
    axis_x: float
    axis_y: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quantify and visualize polarity dynamics from tracked CSV. "
            "Outputs summary tables and figures."
        )
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/all_pv_tracked.csv"),
        help="Tracked CSV input.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/polarity_stats"),
        help="Directory for output tables and figures.",
    )
    p.add_argument(
        "--source-col",
        type=str,
        default=SOURCE_COL,
        help="Source-video column.",
    )
    p.add_argument(
        "--class-filter",
        type=str,
        default="migrating dendritic cell",
        help="Case-insensitive substring filter on class_name.",
    )
    p.add_argument("--min-track-len", type=int, default=8, help="Minimum detections per track to include.")
    p.add_argument(
        "--axis-mode",
        choices=["fixed_x", "fixed_y", "pca_per_source"],
        default="fixed_x",
        help="Polarity axis definition.",
    )
    p.add_argument(
        "--n-time-bins",
        type=int,
        default=25,
        help="Number of normalized-time bins for mean polarity curve.",
    )
    p.add_argument("--max-lag", type=int, default=20, help="Max lag for polarity autocorrelation.")
    p.add_argument(
        "--reversal-eps",
        type=float,
        default=0.1,
        help="Projected-step threshold for sign/reversal counting.",
    )
    p.add_argument(
        "--max-heatmap-tracks",
        type=int,
        default=300,
        help="Max number of tracks shown in polarity heatmap.",
    )
    p.add_argument(
        "--heatmap-resample",
        type=int,
        default=50,
        help="Resampled trajectory length for heatmap (normalized time).",
    )
    return p.parse_args()


def _ensure_columns(df: pd.DataFrame, source_col: str) -> None:
    required = [source_col, "track_id", "frame_index", "x", "y", "class_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def estimate_axis_for_source(source_df: pd.DataFrame, axis_mode: str) -> np.ndarray:
    if axis_mode == "fixed_x":
        return np.array([1.0, 0.0], dtype=float)
    if axis_mode == "fixed_y":
        return np.array([0.0, 1.0], dtype=float)

    # pca_per_source
    dxy_all: list[np.ndarray] = []
    for _, tr in source_df.groupby("track_id", sort=False):
        tr = tr.sort_values("frame_index")
        dx = tr["x"].to_numpy(dtype=float)
        dy = tr["y"].to_numpy(dtype=float)
        if len(dx) < 2:
            continue
        steps = np.stack([np.diff(dx), np.diff(dy)], axis=1)
        if len(steps):
            dxy_all.append(steps)
    if not dxy_all:
        return np.array([1.0, 0.0], dtype=float)

    arr = np.concatenate(dxy_all, axis=0)
    if arr.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=float)
    cov = np.cov(arr.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis = _normalize(axis)
    # Keep sign convention consistent across sources.
    if axis[0] < 0:
        axis = -axis
    return axis


def compute_track_steps(track_df: pd.DataFrame, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = track_df.sort_values("frame_index")
    x = tr["x"].to_numpy(dtype=float)
    y = tr["y"].to_numpy(dtype=float)
    frames = tr["frame_index"].to_numpy(dtype=int)
    if len(x) < 2:
        return np.empty(0), np.empty(0), np.empty(0)

    dx = np.diff(x)
    dy = np.diff(y)
    dframe = np.diff(frames)
    valid = dframe > 0
    if not valid.any():
        return np.empty(0), np.empty(0), np.empty(0)
    dx = dx[valid]
    dy = dy[valid]
    step_norm = np.sqrt(dx * dx + dy * dy)
    proj = dx * axis[0] + dy * axis[1]
    polarity = proj / np.maximum(step_norm, 1e-9)
    return polarity, proj, step_norm


def count_reversals(proj: np.ndarray, eps: float) -> tuple[int, np.ndarray]:
    if len(proj) == 0:
        return 0, np.empty(0)
    s = np.sign(proj)
    s[np.abs(proj) < eps] = 0
    nz_idx = np.where(s != 0)[0]
    if len(nz_idx) < 2:
        return 0, np.empty(0)
    s_nz = s[nz_idx]
    changes = np.where(s_nz[1:] != s_nz[:-1])[0]
    n_rev = int(len(changes))
    if n_rev == 0:
        return 0, np.empty(0)
    # waiting time between reversal events in step units
    rev_positions = nz_idx[changes + 1]
    waits = np.diff(rev_positions).astype(float)
    return n_rev, waits


def build_track_polarity_table(
    df: pd.DataFrame,
    source_col: str,
    axis_mode: str,
    min_track_len: int,
    reversal_eps: float,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray], dict[str, np.ndarray], list[np.ndarray], list[float]]:
    records: list[TrackPolarity] = []
    polarity_by_track: dict[tuple[str, int], np.ndarray] = {}
    axis_by_source: dict[str, np.ndarray] = {}
    all_waits: list[np.ndarray] = []
    all_polarity_values: list[float] = []

    grouped_source = df.groupby(source_col, sort=False)
    for source, src_df in grouped_source:
        axis = estimate_axis_for_source(src_df, axis_mode)
        axis_by_source[str(source)] = axis

        for track_id, tr in src_df.groupby("track_id", sort=False):
            if len(tr) < min_track_len:
                continue
            class_name = str(tr["class_name"].iloc[0])
            pol, proj, step_norm = compute_track_steps(tr, axis)
            if len(pol) == 0:
                continue

            n_rev, waits = count_reversals(proj, reversal_eps)
            if len(waits):
                all_waits.append(waits)

            path_len = float(np.sum(step_norm))
            net_proj = float(np.sum(proj))
            directed_persistence = net_proj / path_len if path_len > 0 else 0.0
            net_disp = float(math.sqrt((tr["x"].iloc[-1] - tr["x"].iloc[0]) ** 2 + (tr["y"].iloc[-1] - tr["y"].iloc[0]) ** 2))

            rec = TrackPolarity(
                source=str(source),
                track_id=int(track_id),
                class_name=class_name,
                n_steps=int(len(pol)),
                n_frames=int(len(tr)),
                mean_polarity=float(np.mean(pol)),
                abs_mean_polarity=float(np.mean(np.abs(pol))),
                directed_persistence=float(directed_persistence),
                net_displacement_px=net_disp,
                path_length_px=path_len,
                reversals=n_rev,
                reversal_rate_per_step=float(n_rev / max(len(pol), 1)),
                axis_x=float(axis[0]),
                axis_y=float(axis[1]),
            )
            records.append(rec)
            polarity_by_track[(str(source), int(track_id))] = pol
            all_polarity_values.extend(pol.tolist())

    metrics_df = pd.DataFrame([r.__dict__ for r in records])
    return metrics_df, polarity_by_track, axis_by_source, all_waits, all_polarity_values


def summarize_polarity_over_time(
    polarity_by_track: dict[tuple[str, int], np.ndarray],
    n_bins: int,
) -> pd.DataFrame:
    if not polarity_by_track:
        return pd.DataFrame(columns=["time_bin", "time_center", "n", "mean", "std", "sem", "ci95_low", "ci95_high"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    values_in_bins: list[list[float]] = [[] for _ in range(n_bins)]

    for pol in polarity_by_track.values():
        n = len(pol)
        if n == 1:
            t_norm = np.array([0.5], dtype=float)
        else:
            t_norm = np.linspace(0.0, 1.0, n, dtype=float)
        idx = np.clip(np.digitize(t_norm, edges, right=True) - 1, 0, n_bins - 1)
        for i, v in zip(idx.tolist(), pol.tolist()):
            values_in_bins[i].append(float(v))

    rows = []
    for i, vals in enumerate(values_in_bins):
        arr = np.asarray(vals, dtype=float)
        n = int(len(arr))
        mean = float(np.mean(arr)) if n else np.nan
        std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 else np.nan
        ci = 1.96 * sem if n > 1 else np.nan
        rows.append(
            dict(
                time_bin=i,
                time_center=float((edges[i] + edges[i + 1]) / 2.0),
                n=n,
                mean=mean,
                std=std,
                sem=sem,
                ci95_low=float(mean - ci) if n > 1 else np.nan,
                ci95_high=float(mean + ci) if n > 1 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def compute_autocorr(
    polarity_by_track: dict[tuple[str, int], np.ndarray],
    max_lag: int,
) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        xvals = []
        yvals = []
        for pol in polarity_by_track.values():
            if len(pol) <= lag:
                continue
            xvals.append(pol[:-lag])
            yvals.append(pol[lag:])
        if not xvals:
            rows.append(dict(lag=lag, n_pairs=0, corr=np.nan))
            continue
        x = np.concatenate(xvals)
        y = np.concatenate(yvals)
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append(dict(lag=lag, n_pairs=int(len(x)), corr=corr))
    return pd.DataFrame(rows)


def plot_hist_mean_polarity(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = metrics_df["mean_polarity"].to_numpy(dtype=float)
    ax.hist(vals, bins=30, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("Mean Track Polarity")
    ax.set_ylabel("Track Count")
    ax.set_title("Distribution of Mean Track Polarity")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_time_curve(time_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = time_df["time_center"].to_numpy(dtype=float)
    y = time_df["mean"].to_numpy(dtype=float)
    lo = time_df["ci95_low"].to_numpy(dtype=float)
    hi = time_df["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, color="#e76f51", linewidth=2)
    mask = np.isfinite(lo) & np.isfinite(hi)
    if mask.any():
        ax.fill_between(x[mask], lo[mask], hi[mask], color="#e76f51", alpha=0.25, linewidth=0)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Normalized Track Time")
    ax.set_ylabel("Instantaneous Polarity")
    ax.set_title("Polarity Dynamics Over Normalized Track Time")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_autocorr(ac_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ac_df["lag"], ac_df["corr"], marker="o", color="#264653")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Polarity Autocorrelation")
    ax.set_title("Polarity Autocorrelation vs Lag")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_reversal_hist(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = metrics_df["reversals"].to_numpy(dtype=int)
    max_v = int(vals.max()) if len(vals) else 0
    bins = np.arange(-0.5, max_v + 1.5, 1.0) if max_v > 0 else np.array([-0.5, 0.5])
    ax.hist(vals, bins=bins, color="#457b9d", edgecolor="white")
    ax.set_xlabel("Reversal Count per Track")
    ax.set_ylabel("Track Count")
    ax.set_title("Track Reversal Count Distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_polarity_heatmap(
    polarity_by_track: dict[tuple[str, int], np.ndarray],
    out: Path,
    max_tracks: int,
    n_resample: int,
) -> None:
    if not polarity_by_track:
        return
    items = sorted(polarity_by_track.items(), key=lambda kv: len(kv[1]), reverse=True)
    items = items[:max_tracks]
    mat = np.full((len(items), n_resample), np.nan, dtype=float)

    for i, (_, pol) in enumerate(items):
        if len(pol) == 1:
            mat[i, :] = pol[0]
            continue
        x_old = np.linspace(0.0, 1.0, len(pol))
        x_new = np.linspace(0.0, 1.0, n_resample)
        mat[i, :] = np.interp(x_new, x_old, pol)

    # order by mean polarity for visual clarity
    ord_idx = np.argsort(np.nanmean(mat, axis=1))
    mat = mat[ord_idx, :]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("Normalized Track Time")
    ax.set_ylabel("Tracks")
    ax.set_title(f"Polarity Heatmap (Top {len(items)} Longest Tracks)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Polarity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    _ensure_columns(df, args.source_col)
    for col in ["track_id", "frame_index", "x", "y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["track_id", "frame_index", "x", "y", "class_name"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)

    cls = args.class_filter.strip().lower()
    if cls:
        mask = df["class_name"].astype(str).str.lower().str.contains(cls, regex=False)
        df = df[mask].copy()

    if len(df) == 0:
        raise RuntimeError("No rows left after class filtering.")

    metrics_df, pol_map, axis_map, all_waits, all_polarity_values = build_track_polarity_table(
        df=df,
        source_col=args.source_col,
        axis_mode=args.axis_mode,
        min_track_len=args.min_track_len,
        reversal_eps=args.reversal_eps,
    )
    if len(metrics_df) == 0:
        raise RuntimeError("No tracks remained after track-length filtering.")

    time_df = summarize_polarity_over_time(pol_map, args.n_time_bins)
    ac_df = compute_autocorr(pol_map, args.max_lag)
    axis_df = pd.DataFrame(
        [dict(source=s, axis_x=float(v[0]), axis_y=float(v[1])) for s, v in axis_map.items()]
    )
    source_summary_df = (
        metrics_df.groupby("source", as_index=False)
        .agg(
            n_tracks=("track_id", "nunique"),
            mean_track_polarity=("mean_polarity", "mean"),
            mean_abs_track_polarity=("abs_mean_polarity", "mean"),
            mean_directed_persistence=("directed_persistence", "mean"),
            mean_reversals=("reversals", "mean"),
        )
        .sort_values("source")
    )

    waits = np.concatenate(all_waits) if all_waits else np.empty(0, dtype=float)
    global_summary = pd.DataFrame(
        [
            dict(
                n_sources=int(metrics_df["source"].nunique()),
                n_tracks=int(metrics_df["track_id"].nunique()),
                mean_track_polarity=float(metrics_df["mean_polarity"].mean()),
                std_track_polarity=float(metrics_df["mean_polarity"].std(ddof=1)),
                mean_abs_track_polarity=float(metrics_df["abs_mean_polarity"].mean()),
                mean_directed_persistence=float(metrics_df["directed_persistence"].mean()),
                mean_reversals=float(metrics_df["reversals"].mean()),
                median_reversals=float(metrics_df["reversals"].median()),
                mean_instant_polarity=float(np.mean(all_polarity_values)),
                std_instant_polarity=float(np.std(all_polarity_values, ddof=1)),
                mean_reversal_wait_steps=float(np.mean(waits)) if len(waits) else np.nan,
                median_reversal_wait_steps=float(np.median(waits)) if len(waits) else np.nan,
            )
        ]
    )

    # Save tables
    metrics_csv = args.output_dir / "polarity_track_metrics.csv"
    time_csv = args.output_dir / "polarity_time_profile.csv"
    ac_csv = args.output_dir / "polarity_autocorr.csv"
    source_csv = args.output_dir / "polarity_source_summary.csv"
    axis_csv = args.output_dir / "polarity_axes.csv"
    summary_csv = args.output_dir / "polarity_global_summary.csv"
    waits_csv = args.output_dir / "polarity_reversal_waits.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    time_df.to_csv(time_csv, index=False)
    ac_df.to_csv(ac_csv, index=False)
    source_summary_df.to_csv(source_csv, index=False)
    axis_df.to_csv(axis_csv, index=False)
    global_summary.to_csv(summary_csv, index=False)
    pd.DataFrame({"reversal_wait_steps": waits}).to_csv(waits_csv, index=False)

    # Save figures
    plot_hist_mean_polarity(metrics_df, args.output_dir / "polarity_hist_mean_track.png")
    plot_time_curve(time_df, args.output_dir / "polarity_time_profile.png")
    plot_autocorr(ac_df, args.output_dir / "polarity_autocorr.png")
    plot_reversal_hist(metrics_df, args.output_dir / "polarity_reversal_count_hist.png")
    plot_polarity_heatmap(
        pol_map,
        args.output_dir / "polarity_heatmap.png",
        max_tracks=args.max_heatmap_tracks,
        n_resample=args.heatmap_resample,
    )

    print(f"Input rows after class filter: {len(df)}")
    print(f"Analyzed tracks: {metrics_df['track_id'].nunique()}")
    print(f"Output directory: {args.output_dir}")
    print("Saved tables:")
    for p in [metrics_csv, time_csv, ac_csv, source_csv, axis_csv, summary_csv, waits_csv]:
        print(f"  - {p}")
    print("Saved figures:")
    for p in [
        args.output_dir / "polarity_hist_mean_track.png",
        args.output_dir / "polarity_time_profile.png",
        args.output_dir / "polarity_autocorr.png",
        args.output_dir / "polarity_reversal_count_hist.png",
        args.output_dir / "polarity_heatmap.png",
    ]:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
