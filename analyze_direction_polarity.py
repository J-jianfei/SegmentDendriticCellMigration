#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


@dataclass
class DirectionTrackMetrics:
    source: str
    track_id: int
    class_name: str
    n_steps: int
    path_length_px: float
    net_displacement_px: float
    mean_step_px: float
    resultant_length: float
    circular_variance: float
    mean_cos_turn: float
    reversal_fraction: float
    mean_initial_alignment: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Direction-based polarity analysis: polarity is movement direction (unit velocity). "
            "Computes directional autocorrelation and time-dependent direction memory."
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
        default=Path("/mnt/Files/Dendritic_Cells/Migration/direction_polarity_stats"),
        help="Output directory for tables and plots.",
    )
    p.add_argument("--source-col", type=str, default=SOURCE_COL, help="Source-video column name.")
    p.add_argument(
        "--class-filter",
        type=str,
        default="migrating dendritic cell",
        help="Case-insensitive class_name substring filter.",
    )
    p.add_argument("--min-track-len", type=int, default=8, help="Minimum detections per track.")
    p.add_argument("--max-lag", type=int, default=20, help="Maximum lag (in steps) for autocorrelation.")
    p.add_argument("--n-time-bins", type=int, default=25, help="Bins for normalized-time profile.")
    p.add_argument(
        "--max-heatmap-tracks",
        type=int,
        default=300,
        help="Max tracks shown in heatmap.",
    )
    p.add_argument(
        "--heatmap-resample",
        type=int,
        default=50,
        help="Resampled points for heatmap/time profile visualization.",
    )
    p.add_argument(
        "--step-eps",
        type=float,
        default=5.0,
        help="Minimum step size (px) to treat as valid direction step (default 5 px to reduce jitter bias).",
    )
    p.add_argument(
        "--direction-frame-lag",
        type=int,
        default=2,
        help=(
            "Frame lag used to compute direction vectors. "
            "Uses only pairs with exact frame separation equal to this lag."
        ),
    )
    p.add_argument(
        "--wt-regex",
        type=str,
        default=r"^wt(?:_|\d|$)",
        help="Regex (applied to source filename stem) used to label WT condition.",
    )
    p.add_argument(
        "--myoko-regex",
        type=str,
        default=r"myoko",
        help="Regex (applied to source filename stem) used to label MyoKO condition.",
    )
    p.add_argument(
        "--unknown-condition-label",
        type=str,
        default="other",
        help="Label used when source filename does not match WT or MyoKO regex.",
    )
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, source_col: str) -> None:
    required = [source_col, "track_id", "frame_index", "x", "y", "class_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_track_directions(
    track_df: pd.DataFrame, step_eps: float, direction_frame_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    tr = track_df.sort_values("frame_index")
    x = tr["x"].to_numpy(dtype=float)
    y = tr["y"].to_numpy(dtype=float)
    f = tr["frame_index"].to_numpy(dtype=int)
    lag = max(1, int(direction_frame_lag))
    if len(x) < lag + 1:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)

    # Use exact frame lag to avoid mixed time steps in direction autocorrelation.
    idx = [i for i in range(len(f) - lag) if (f[i + lag] - f[i]) == lag]
    if not idx:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    idx_arr = np.asarray(idx, dtype=int)
    dx = x[idx_arr + lag] - x[idx_arr]
    dy = y[idx_arr + lag] - y[idx_arr]
    step = np.sqrt(dx * dx + dy * dy)
    valid2 = step > step_eps
    if not valid2.any():
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    dx = dx[valid2]
    dy = dy[valid2]
    step = step[valid2]
    u = np.stack([dx / step, dy / step], axis=1)
    return u, step


def infer_condition(
    source: str,
    wt_pattern: re.Pattern[str],
    myoko_pattern: re.Pattern[str],
    unknown_label: str,
) -> str:
    stem = Path(str(source)).stem.lower()
    if myoko_pattern.search(stem):
        return "MyoKO"
    if wt_pattern.search(stem):
        return "WT"
    return unknown_label


def track_metrics_from_u(
    source: str,
    track_id: int,
    class_name: str,
    u: np.ndarray,
    step: np.ndarray,
    xy_start: tuple[float, float],
    xy_end: tuple[float, float],
) -> DirectionTrackMetrics:
    n = len(u)
    res_vec = np.mean(u, axis=0)
    resultant = float(np.linalg.norm(res_vec))
    circ_var = float(1.0 - resultant)

    if n >= 2:
        dots = np.sum(u[:-1] * u[1:], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        mean_cos_turn = float(np.mean(dots))
        reversal_fraction = float(np.mean(dots < 0))
    else:
        mean_cos_turn = np.nan
        reversal_fraction = np.nan

    u0 = u[0]
    init_align = np.sum(u * u0[None, :], axis=1)
    init_align_mean = float(np.mean(init_align))

    path_len = float(np.sum(step))
    net_disp = float(math.hypot(xy_end[0] - xy_start[0], xy_end[1] - xy_start[1]))
    mean_step = float(np.mean(step))

    return DirectionTrackMetrics(
        source=source,
        track_id=track_id,
        class_name=class_name,
        n_steps=n,
        path_length_px=path_len,
        net_displacement_px=net_disp,
        mean_step_px=mean_step,
        resultant_length=resultant,
        circular_variance=circ_var,
        mean_cos_turn=mean_cos_turn,
        reversal_fraction=reversal_fraction,
        mean_initial_alignment=init_align_mean,
    )


def summarize_direction_time_profile(
    init_align_by_track: dict[tuple[str, int], np.ndarray],
    n_bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    vals = [[] for _ in range(n_bins)]

    for arr in init_align_by_track.values():
        n = len(arr)
        if n == 0:
            continue
        t = np.linspace(0.0, 1.0, n)
        idx = np.clip(np.digitize(t, edges, right=True) - 1, 0, n_bins - 1)
        for i, v in zip(idx.tolist(), arr.tolist()):
            vals[i].append(float(v))

    rows = []
    for i, v in enumerate(vals):
        arr = np.asarray(v, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr)) if n else np.nan
        std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 else np.nan
        ci = 1.96 * sem if n > 1 else np.nan
        rows.append(
            dict(
                time_bin=i,
                time_center=float((edges[i] + edges[i + 1]) / 2),
                n=n,
                mean=mean,
                std=std,
                sem=sem,
                ci95_low=float(mean - ci) if n > 1 else np.nan,
                ci95_high=float(mean + ci) if n > 1 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def summarize_direction_autocorr(
    u_by_track: dict[tuple[str, int], np.ndarray],
    max_lag: int,
) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        vals = []
        for u in u_by_track.values():
            if len(u) <= lag:
                continue
            dots = np.sum(u[:-lag] * u[lag:], axis=1)
            vals.extend(dots.tolist())
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr)) if n else np.nan
        std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 else np.nan
        ci = 1.96 * sem if n > 1 else np.nan
        rows.append(
            dict(
                lag=lag,
                n_pairs=n,
                mean=mean,
                std=std,
                sem=sem,
                ci95_low=float(mean - ci) if n > 1 else np.nan,
                ci95_high=float(mean + ci) if n > 1 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def summarize_direction_autocorr_by_condition(
    u_by_track: dict[tuple[str, int], np.ndarray],
    condition_by_track: dict[tuple[str, int], str],
    max_lag: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    conditions = sorted(set(condition_by_track.values()))
    for cond in conditions:
        subset = {k: u for k, u in u_by_track.items() if condition_by_track.get(k) == cond}
        if not subset:
            continue
        ac_df = summarize_direction_autocorr(subset, max_lag=max_lag)
        ac_df.insert(0, "condition", cond)
        ac_df.insert(1, "n_tracks", len(subset))
        rows.append(ac_df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["condition", "n_tracks", "lag", "n_pairs", "mean", "std", "sem", "ci95_low", "ci95_high"]
    )


def summarize_direction_time_profile_by_condition(
    init_align_by_track: dict[tuple[str, int], np.ndarray],
    condition_by_track: dict[tuple[str, int], str],
    n_bins: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    conditions = sorted(set(condition_by_track.values()))
    for cond in conditions:
        subset = {k: arr for k, arr in init_align_by_track.items() if condition_by_track.get(k) == cond}
        if not subset:
            continue
        time_df = summarize_direction_time_profile(subset, n_bins=n_bins)
        time_df.insert(0, "condition", cond)
        time_df.insert(1, "n_tracks", len(subset))
        rows.append(time_df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        columns=["condition", "n_tracks", "time_bin", "time_center", "n", "mean", "std", "sem", "ci95_low", "ci95_high"]
    )


def plot_autocorr(ac_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = ac_df["lag"].to_numpy(dtype=float)
    y = ac_df["mean"].to_numpy(dtype=float)
    lo = ac_df["ci95_low"].to_numpy(dtype=float)
    hi = ac_df["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", color="#264653", linewidth=2)
    m = np.isfinite(lo) & np.isfinite(hi)
    if m.any():
        ax.fill_between(x[m], lo[m], hi[m], color="#264653", alpha=0.2, linewidth=0)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel(r"$\langle \hat{u}(t)\cdot\hat{u}(t+\tau)\rangle$")
    ax.set_title("Directional Autocorrelation")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_time_profile(time_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = time_df["time_center"].to_numpy(dtype=float)
    y = time_df["mean"].to_numpy(dtype=float)
    lo = time_df["ci95_low"].to_numpy(dtype=float)
    hi = time_df["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, color="#e76f51", linewidth=2)
    m = np.isfinite(lo) & np.isfinite(hi)
    if m.any():
        ax.fill_between(x[m], lo[m], hi[m], color="#e76f51", alpha=0.25, linewidth=0)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Normalized Track Time")
    ax.set_ylabel(r"$\hat{u}(t)\cdot\hat{u}(0)$")
    ax.set_title("Direction Memory Over Time")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_autocorr_by_condition(ac_cond_df: pd.DataFrame, out: Path) -> None:
    if ac_cond_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"WT": "#2a9d8f", "MyoKO": "#e76f51"}
    for cond, part in ac_cond_df.groupby("condition", sort=True):
        part = part.sort_values("lag")
        x = part["lag"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        lo = part["ci95_low"].to_numpy(dtype=float)
        hi = part["ci95_high"].to_numpy(dtype=float)
        color = palette.get(str(cond), None)
        ax.plot(x, y, marker="o", linewidth=2, label=str(cond), color=color)
        m = np.isfinite(lo) & np.isfinite(hi)
        if m.any():
            ax.fill_between(x[m], lo[m], hi[m], alpha=0.15, linewidth=0, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel(r"$\langle \hat{u}(t)\cdot\hat{u}(t+\tau)\rangle$")
    ax.set_title("Directional Autocorrelation by Condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_time_profile_by_condition(time_cond_df: pd.DataFrame, out: Path) -> None:
    if time_cond_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"WT": "#2a9d8f", "MyoKO": "#e76f51"}
    for cond, part in time_cond_df.groupby("condition", sort=True):
        part = part.sort_values("time_center")
        x = part["time_center"].to_numpy(dtype=float)
        y = part["mean"].to_numpy(dtype=float)
        lo = part["ci95_low"].to_numpy(dtype=float)
        hi = part["ci95_high"].to_numpy(dtype=float)
        color = palette.get(str(cond), None)
        ax.plot(x, y, linewidth=2, label=str(cond), color=color)
        m = np.isfinite(lo) & np.isfinite(hi)
        if m.any():
            ax.fill_between(x[m], lo[m], hi[m], alpha=0.15, linewidth=0, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Normalized Track Time")
    ax.set_ylabel(r"$\hat{u}(t)\cdot\hat{u}(0)$")
    ax.set_title("Direction Memory Over Time by Condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_resultant_hist(metrics_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = metrics_df["resultant_length"].to_numpy(dtype=float)
    ax.hist(vals, bins=30, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("Resultant Length (Directionality)")
    ax.set_ylabel("Track Count")
    ax.set_title("Track Directionality Distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_turning_angle_hist(turn_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = turn_df["turn_angle_deg"].to_numpy(dtype=float)
    ax.hist(vals, bins=np.linspace(0, 180, 37), color="#457b9d", edgecolor="white")
    ax.set_xlabel("Turning Angle (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Turning Angle Distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_memory_heatmap(
    init_align_by_track: dict[tuple[str, int], np.ndarray],
    out: Path,
    max_tracks: int,
    n_resample: int,
) -> None:
    if not init_align_by_track:
        return
    items = sorted(init_align_by_track.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_tracks]
    mat = np.full((len(items), n_resample), np.nan, dtype=float)
    for i, (_, arr) in enumerate(items):
        if len(arr) == 1:
            mat[i, :] = arr[0]
        else:
            x_old = np.linspace(0.0, 1.0, len(arr))
            x_new = np.linspace(0.0, 1.0, n_resample)
            mat[i, :] = np.interp(x_new, x_old, arr)
    ord_idx = np.argsort(np.nanmean(mat, axis=1))
    mat = mat[ord_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("Normalized Track Time")
    ax.set_ylabel("Tracks")
    ax.set_title(f"Direction Memory Heatmap (Top {len(items)} Longest Tracks)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\hat{u}(t)\cdot\hat{u}(0)$")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    ensure_columns(df, args.source_col)
    for c in ["track_id", "frame_index", "x", "y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["track_id", "frame_index", "x", "y", "class_name"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)

    cf = args.class_filter.strip().lower()
    if cf:
        df = df[df["class_name"].astype(str).str.lower().str.contains(cf, regex=False)].copy()
    if len(df) == 0:
        raise RuntimeError("No rows left after filtering.")

    records: list[DirectionTrackMetrics] = []
    u_by_track: dict[tuple[str, int], np.ndarray] = {}
    init_align_by_track: dict[tuple[str, int], np.ndarray] = {}
    condition_by_track: dict[tuple[str, int], str] = {}
    turn_rows: list[dict[str, float]] = []
    wt_pattern = re.compile(args.wt_regex, flags=re.IGNORECASE)
    myoko_pattern = re.compile(args.myoko_regex, flags=re.IGNORECASE)

    for (source, track_id), tr in df.groupby([args.source_col, "track_id"], sort=False):
        if len(tr) < args.min_track_len:
            continue
        source_str = str(source)
        condition = infer_condition(
            source=source_str,
            wt_pattern=wt_pattern,
            myoko_pattern=myoko_pattern,
            unknown_label=args.unknown_condition_label,
        )
        class_name = str(tr["class_name"].iloc[0])
        u, step = compute_track_directions(
            tr,
            step_eps=args.step_eps,
            direction_frame_lag=args.direction_frame_lag,
        )
        if len(u) < max(2, args.min_track_len - 1):
            continue

        u0 = u[0]
        init_align = np.sum(u * u0[None, :], axis=1)
        key = (source_str, int(track_id))
        init_align_by_track[key] = init_align
        u_by_track[key] = u
        condition_by_track[key] = condition

        if len(u) >= 2:
            dots = np.sum(u[:-1] * u[1:], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            for a in angles.tolist():
                turn_rows.append(
                    dict(
                        source=source_str,
                        track_id=int(track_id),
                        condition=condition,
                        turn_angle_deg=float(a),
                    )
                )

        tr_sorted = tr.sort_values("frame_index")
        rec = track_metrics_from_u(
            source=source_str,
            track_id=int(track_id),
            class_name=class_name,
            u=u,
            step=step,
            xy_start=(float(tr_sorted["x"].iloc[0]), float(tr_sorted["y"].iloc[0])),
            xy_end=(float(tr_sorted["x"].iloc[-1]), float(tr_sorted["y"].iloc[-1])),
        )
        records.append(rec)

    if not records:
        raise RuntimeError("No valid tracks remained for direction analysis.")

    metrics_df = pd.DataFrame([r.__dict__ for r in records])
    metrics_df["condition"] = metrics_df["source"].map(
        lambda s: infer_condition(
            source=str(s),
            wt_pattern=wt_pattern,
            myoko_pattern=myoko_pattern,
            unknown_label=args.unknown_condition_label,
        )
    )
    time_df = summarize_direction_time_profile(init_align_by_track, args.n_time_bins)
    ac_df = summarize_direction_autocorr(u_by_track, args.max_lag)
    cond_time_df = summarize_direction_time_profile_by_condition(
        init_align_by_track=init_align_by_track,
        condition_by_track=condition_by_track,
        n_bins=args.n_time_bins,
    )
    cond_ac_df = summarize_direction_autocorr_by_condition(
        u_by_track=u_by_track,
        condition_by_track=condition_by_track,
        max_lag=args.max_lag,
    )
    turn_df = pd.DataFrame(turn_rows)

    global_summary = pd.DataFrame(
        [
            dict(
                n_sources=int(metrics_df["source"].nunique()),
                n_tracks=int(metrics_df["track_id"].nunique()),
                mean_resultant_length=float(metrics_df["resultant_length"].mean()),
                median_resultant_length=float(metrics_df["resultant_length"].median()),
                mean_circular_variance=float(metrics_df["circular_variance"].mean()),
                mean_reversal_fraction=float(metrics_df["reversal_fraction"].mean(skipna=True)),
                mean_cos_turn=float(metrics_df["mean_cos_turn"].mean(skipna=True)),
                mean_initial_alignment=float(metrics_df["mean_initial_alignment"].mean()),
            )
        ]
    )

    source_summary_df = (
        metrics_df.groupby("source", as_index=False)
        .agg(
            condition=("condition", "first"),
            n_tracks=("track_id", "nunique"),
            mean_resultant_length=("resultant_length", "mean"),
            mean_circular_variance=("circular_variance", "mean"),
            mean_reversal_fraction=("reversal_fraction", "mean"),
            mean_initial_alignment=("mean_initial_alignment", "mean"),
        )
        .sort_values("source")
    )
    condition_summary_df = (
        metrics_df.groupby("condition", as_index=False)
        .agg(
            n_sources=("source", "nunique"),
            n_tracks=("track_id", "count"),
            mean_resultant_length=("resultant_length", "mean"),
            median_resultant_length=("resultant_length", "median"),
            mean_circular_variance=("circular_variance", "mean"),
            mean_reversal_fraction=("reversal_fraction", "mean"),
            mean_cos_turn=("mean_cos_turn", "mean"),
            mean_initial_alignment=("mean_initial_alignment", "mean"),
            mean_step_px=("mean_step_px", "mean"),
        )
        .sort_values("condition")
    )

    # save tables
    metrics_csv = args.output_dir / "direction_track_metrics.csv"
    ac_csv = args.output_dir / "direction_autocorr.csv"
    time_csv = args.output_dir / "direction_memory_time_profile.csv"
    cond_ac_csv = args.output_dir / "direction_condition_autocorr.csv"
    cond_time_csv = args.output_dir / "direction_condition_memory_time_profile.csv"
    turn_csv = args.output_dir / "direction_turning_angles.csv"
    source_csv = args.output_dir / "direction_source_summary.csv"
    condition_csv = args.output_dir / "direction_condition_summary.csv"
    global_csv = args.output_dir / "direction_global_summary.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    ac_df.to_csv(ac_csv, index=False)
    time_df.to_csv(time_csv, index=False)
    cond_ac_df.to_csv(cond_ac_csv, index=False)
    cond_time_df.to_csv(cond_time_csv, index=False)
    turn_df.to_csv(turn_csv, index=False)
    source_summary_df.to_csv(source_csv, index=False)
    condition_summary_df.to_csv(condition_csv, index=False)
    global_summary.to_csv(global_csv, index=False)

    # plots
    plot_autocorr(ac_df, args.output_dir / "direction_autocorr.png")
    plot_time_profile(time_df, args.output_dir / "direction_memory_time_profile.png")
    plot_autocorr_by_condition(cond_ac_df, args.output_dir / "direction_autocorr_by_condition.png")
    plot_time_profile_by_condition(cond_time_df, args.output_dir / "direction_memory_time_profile_by_condition.png")
    plot_resultant_hist(metrics_df, args.output_dir / "directionality_resultant_hist.png")
    plot_turning_angle_hist(turn_df, args.output_dir / "direction_turning_angle_hist.png")
    plot_memory_heatmap(
        init_align_by_track,
        args.output_dir / "direction_memory_heatmap.png",
        max_tracks=args.max_heatmap_tracks,
        n_resample=args.heatmap_resample,
    )

    print(f"Input rows after class filter: {len(df)}")
    print(f"Analyzed tracks: {metrics_df['track_id'].nunique()}")
    print(f"Output directory: {args.output_dir}")
    print("Saved tables:")
    for p in [metrics_csv, ac_csv, time_csv, cond_ac_csv, cond_time_csv, turn_csv, source_csv, condition_csv, global_csv]:
        print(f"  - {p}")
    print("Saved figures:")
    for p in [
        args.output_dir / "direction_autocorr.png",
        args.output_dir / "direction_memory_time_profile.png",
        args.output_dir / "direction_autocorr_by_condition.png",
        args.output_dir / "direction_memory_time_profile_by_condition.png",
        args.output_dir / "directionality_resultant_hist.png",
        args.output_dir / "direction_turning_angle_hist.png",
        args.output_dir / "direction_memory_heatmap.png",
    ]:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
