#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit a WT-only stochastic polarity process and simulate comparison statistics "
            "(directional autocorrelation, turning-angle distribution, and track directionality)."
        )
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/all_pv_tracked.csv"),
        help="Tracked CSV input path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/wt_polarity_model"),
        help="Output folder for fitted parameters, tables, and plots.",
    )
    p.add_argument("--source-col", type=str, default=SOURCE_COL, help="Source video path column.")
    p.add_argument(
        "--class-filter",
        type=str,
        default="migrating dendritic cell",
        help="Case-insensitive class_name substring filter.",
    )
    p.add_argument(
        "--wt-regex",
        type=str,
        default=r"^wt(?:_|\d|$)",
        help="Regex on source filename stem used to select WT tracks.",
    )
    p.add_argument("--min-track-len", type=int, default=8, help="Minimum detections per track.")
    p.add_argument(
        "--direction-frame-lag",
        type=int,
        default=2,
        help="Frame lag used to compute direction vectors from track positions.",
    )
    p.add_argument("--step-eps", type=float, default=5.0, help="Minimum displacement (px) for valid direction step.")
    p.add_argument("--max-lag", type=int, default=20, help="Maximum lag for directional autocorrelation.")
    p.add_argument(
        "--n-sim-tracks",
        type=int,
        default=0,
        help="Number of simulated tracks. 0 uses the same count as observed WT tracks.",
    )
    p.add_argument(
        "--fit-search-iter",
        type=int,
        default=220,
        help="Random-search iterations used to tune direction-process parameters.",
    )
    p.add_argument(
        "--fit-search-seed",
        type=int,
        default=23,
        help="Seed for direction-parameter random search.",
    )
    p.add_argument(
        "--score-resultant-weight",
        type=float,
        default=0.35,
        help="Objective weight for matching mean resultant length.",
    )
    p.add_argument(
        "--score-reversal-weight",
        type=float,
        default=0.10,
        help="Objective weight for matching mean reversal fraction.",
    )
    p.add_argument("--seed", type=int, default=7, help="Random seed for simulation.")
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, source_col: str) -> None:
    required = [source_col, "track_id", "frame_index", "x", "y", "class_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def compute_track_directions(
    track_df: pd.DataFrame,
    step_eps: float,
    direction_frame_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    tr = track_df.sort_values("frame_index")
    x = tr["x"].to_numpy(dtype=float)
    y = tr["y"].to_numpy(dtype=float)
    f = tr["frame_index"].to_numpy(dtype=int)
    lag = max(1, int(direction_frame_lag))
    if len(x) < lag + 1:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)

    idx = [i for i in range(len(f) - lag) if (f[i + lag] - f[i]) == lag]
    if not idx:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    idx_arr = np.asarray(idx, dtype=int)
    dx = x[idx_arr + lag] - x[idx_arr]
    dy = y[idx_arr + lag] - y[idx_arr]
    step = np.sqrt(dx * dx + dy * dy)
    valid = step > step_eps
    if not valid.any():
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    dx = dx[valid]
    dy = dy[valid]
    step = step[valid]
    u = np.stack([dx / step, dy / step], axis=1)
    return u, step


def signed_turn_angles(u: np.ndarray) -> np.ndarray:
    if len(u) < 2:
        return np.empty(0, dtype=float)
    ux = u[:-1, 0]
    uy = u[:-1, 1]
    vx = u[1:, 0]
    vy = u[1:, 1]
    dot = np.clip(ux * vx + uy * vy, -1.0, 1.0)
    cross = ux * vy - uy * vx
    return np.arctan2(cross, dot)


def mean_resultant_length(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return 0.0
    z = np.exp(1j * angles)
    return float(np.abs(np.mean(z)))


def a1inv(r: float) -> float:
    r = max(1e-9, min(0.999999, float(r)))
    if r < 0.53:
        return 2.0 * r + r**3 + 5.0 * (r**5) / 6.0
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    return 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)


def summarize_autocorr(u_tracks: list[np.ndarray], max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for lag in range(1, max_lag + 1):
        vals: list[float] = []
        for u in u_tracks:
            if len(u) <= lag:
                continue
            dots = np.sum(u[:-lag] * u[lag:], axis=1)
            vals.extend(dots.tolist())
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr)) if n else np.nan
        std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 else np.nan
        rows.append(dict(lag=lag, n_pairs=n, mean=mean, std=std, sem=sem))
    return pd.DataFrame(rows)


def build_observed_tracks(
    df: pd.DataFrame,
    source_col: str,
    wt_pattern: re.Pattern[str],
    class_filter: str,
    min_track_len: int,
    step_eps: float,
    direction_frame_lag: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    cf = class_filter.strip().lower()
    if cf:
        df = df[df["class_name"].astype(str).str.lower().str.contains(cf, regex=False)].copy()
    if len(df) == 0:
        return [], [], []

    u_tracks: list[np.ndarray] = []
    steps_tracks: list[np.ndarray] = []
    turn_tracks: list[np.ndarray] = []
    for (source, _track_id), tr in df.groupby([source_col, "track_id"], sort=False):
        stem = Path(str(source)).stem.lower()
        if not wt_pattern.search(stem):
            continue
        if len(tr) < min_track_len:
            continue
        u, step = compute_track_directions(
            tr,
            step_eps=step_eps,
            direction_frame_lag=direction_frame_lag,
        )
        if len(u) < max(2, min_track_len - 1):
            continue
        turns = signed_turn_angles(u)
        u_tracks.append(u)
        steps_tracks.append(step)
        turn_tracks.append(turns)
    return u_tracks, steps_tracks, turn_tracks


def fit_wt_process_moment_init(turn_tracks: list[np.ndarray], steps_tracks: list[np.ndarray]) -> dict[str, float]:
    turns = np.concatenate([t for t in turn_tracks if len(t) > 0]) if turn_tracks else np.empty(0, dtype=float)
    if len(turns) == 0:
        raise RuntimeError("No WT turning-angle samples available for fitting.")
    rev_mask = np.cos(turns) < 0.0
    p_rev = float(np.mean(rev_mask))

    fwd = turns[~rev_mask]
    if len(fwd) == 0:
        raise RuntimeError("No forward-turn samples available for fitting.")
    r_fwd = mean_resultant_length(fwd)
    kappa_fwd = float(a1inv(r_fwd))

    rev = turns[rev_mask]
    if len(rev) > 0:
        rev_centered = np.where(rev >= 0.0, rev - np.pi, rev + np.pi)
        r_rev = mean_resultant_length(rev_centered)
        kappa_rev = float(a1inv(r_rev))
    else:
        r_rev = 0.0
        kappa_rev = max(0.5, 0.5 * kappa_fwd)

    all_steps = np.concatenate([s for s in steps_tracks if len(s) > 0]) if steps_tracks else np.empty(0, dtype=float)
    if len(all_steps) == 0:
        raise RuntimeError("No WT step-size samples available for fitting.")
    log_steps = np.log(np.clip(all_steps, 1e-6, None))
    step_log_mu = float(np.mean(log_steps))
    step_log_sigma = float(np.std(log_steps, ddof=1)) if len(log_steps) > 1 else 0.1

    return dict(
        p_rev=p_rev,
        kappa_fwd=kappa_fwd,
        kappa_rev=kappa_rev,
        bias_mix=0.0,
        r_fwd=r_fwd,
        r_rev=r_rev,
        step_log_mu=step_log_mu,
        step_log_sigma=step_log_sigma,
        n_turn_samples=int(len(turns)),
        n_step_samples=int(len(all_steps)),
    )


def simulate_direction_tracks(
    n_tracks: int,
    lengths: np.ndarray,
    params: dict[str, float],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    u_tracks: list[np.ndarray] = []
    turn_tracks: list[np.ndarray] = []
    if len(lengths) == 0 or n_tracks <= 0:
        return u_tracks, turn_tracks

    lens = rng.choice(lengths, size=n_tracks, replace=True)
    bias_mix = float(params.get("bias_mix", 0.0))
    bias_mix = min(0.95, max(0.0, bias_mix))

    for n in lens.tolist():
        n_int = int(max(2, n))
        theta = float(rng.uniform(-np.pi, np.pi))
        psi = float(rng.uniform(-np.pi, np.pi))
        bias_dir = np.array([math.cos(psi), math.sin(psi)], dtype=float)
        u = np.zeros((n_int, 2), dtype=float)

        for t in range(n_int):
            base_u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
            if bias_mix > 0.0:
                mixed = (1.0 - bias_mix) * base_u + bias_mix * bias_dir
                nrm = float(np.linalg.norm(mixed))
                if nrm > 0:
                    mixed /= nrm
                u[t] = mixed
            else:
                u[t] = base_u

            if t < n_int - 1:
                if rng.random() < params["p_rev"]:
                    base = np.pi if rng.random() < 0.5 else -np.pi
                    delta = base + float(rng.vonmises(mu=0.0, kappa=params["kappa_rev"]))
                else:
                    delta = float(rng.vonmises(mu=0.0, kappa=params["kappa_fwd"]))
                delta = float(wrap_pi(delta))
                theta = float(wrap_pi(theta + delta))

        turns = signed_turn_angles(u)
        u_tracks.append(u)
        turn_tracks.append(turns)
    return u_tracks, turn_tracks


def score_direction_model(
    obs_u_tracks: list[np.ndarray],
    obs_turn_tracks: list[np.ndarray],
    sim_u_tracks: list[np.ndarray],
    sim_turn_tracks: list[np.ndarray],
    max_lag: int,
    score_resultant_weight: float,
    score_reversal_weight: float,
) -> dict[str, float]:
    obs_ac = summarize_autocorr(obs_u_tracks, max_lag=max_lag)
    sim_ac = summarize_autocorr(sim_u_tracks, max_lag=max_lag)
    ac_cmp = obs_ac.merge(sim_ac, on="lag", suffixes=("_obs", "_sim"))
    rmse = float(np.sqrt(np.mean((ac_cmp["mean_obs"].to_numpy() - ac_cmp["mean_sim"].to_numpy()) ** 2)))

    obs_metrics = track_metrics(obs_u_tracks, obs_turn_tracks)
    sim_metrics = track_metrics(sim_u_tracks, sim_turn_tracks)
    obs_r = float(obs_metrics["resultant_length"].mean())
    sim_r = float(sim_metrics["resultant_length"].mean())
    obs_rev = float(obs_metrics["reversal_fraction"].mean(skipna=True))
    sim_rev = float(sim_metrics["reversal_fraction"].mean(skipna=True))
    score = rmse + score_resultant_weight * abs(sim_r - obs_r) + score_reversal_weight * abs(sim_rev - obs_rev)

    return dict(
        score=score,
        ac_rmse=rmse,
        obs_mean_resultant_length=obs_r,
        sim_mean_resultant_length=sim_r,
        obs_mean_reversal_fraction=obs_rev,
        sim_mean_reversal_fraction=sim_rev,
    )


def fit_direction_params_by_search(
    obs_u_tracks: list[np.ndarray],
    obs_turn_tracks: list[np.ndarray],
    init_params: dict[str, float],
    max_lag: int,
    n_sim_tracks: int,
    n_iter: int,
    seed: int,
    score_resultant_weight: float,
    score_reversal_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    lengths = np.asarray([len(u) for u in obs_u_tracks], dtype=int)
    if len(lengths) == 0:
        raise RuntimeError("No observed WT direction tracks available for fitting.")

    rng = np.random.default_rng(seed)
    candidates: list[dict[str, float]] = []

    # include moment-init candidate first
    candidates.append(
        dict(
            p_rev=float(init_params["p_rev"]),
            kappa_fwd=float(init_params["kappa_fwd"]),
            kappa_rev=float(init_params["kappa_rev"]),
            bias_mix=float(init_params.get("bias_mix", 0.0)),
        )
    )

    for i in range(max(0, int(n_iter))):
        if i < int(0.35 * n_iter):
            # Local perturbation around moment estimate
            p_rev = float(np.clip(rng.normal(init_params["p_rev"], 0.05), 0.02, 0.35))
            kappa_fwd = float(np.clip(np.exp(rng.normal(np.log(max(1e-4, init_params["kappa_fwd"])), 0.45)), 1.2, 25.0))
            kappa_rev = float(np.clip(np.exp(rng.normal(np.log(max(1e-4, init_params["kappa_rev"])), 0.50)), 0.25, 8.0))
            bias_mix = float(np.clip(rng.normal(0.35, 0.18), 0.0, 0.65))
        else:
            # Global search
            p_rev = float(rng.uniform(0.03, 0.30))
            kappa_fwd = float(np.exp(rng.uniform(np.log(1.2), np.log(25.0))))
            kappa_rev = float(np.exp(rng.uniform(np.log(0.25), np.log(8.0))))
            bias_mix = float(rng.uniform(0.0, 0.65))
        candidates.append(dict(p_rev=p_rev, kappa_fwd=kappa_fwd, kappa_rev=kappa_rev, bias_mix=bias_mix))

    best_params: dict[str, float] | None = None
    best_diag: dict[str, float] | None = None
    for i, cand in enumerate(candidates):
        sim_rng = np.random.default_rng(seed + 1000 + i)
        sim_u_tracks, sim_turn_tracks = simulate_direction_tracks(
            n_tracks=n_sim_tracks,
            lengths=lengths,
            params=cand,
            rng=sim_rng,
        )
        diag = score_direction_model(
            obs_u_tracks=obs_u_tracks,
            obs_turn_tracks=obs_turn_tracks,
            sim_u_tracks=sim_u_tracks,
            sim_turn_tracks=sim_turn_tracks,
            max_lag=max_lag,
            score_resultant_weight=score_resultant_weight,
            score_reversal_weight=score_reversal_weight,
        )
        if best_diag is None or float(diag["score"]) < float(best_diag["score"]):
            best_diag = diag
            best_params = cand

    if best_params is None or best_diag is None:
        raise RuntimeError("Direction-parameter fitting failed to produce a candidate.")
    return best_params, best_diag


def track_metrics(u_tracks: list[np.ndarray], turn_tracks: list[np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i, (u, turns) in enumerate(zip(u_tracks, turn_tracks), start=1):
        if len(u) == 0:
            continue
        res = float(np.linalg.norm(np.mean(u, axis=0)))
        if len(turns) > 0:
            dots = np.cos(turns)
            mean_cos_turn = float(np.mean(dots))
            reversal_fraction = float(np.mean(dots < 0.0))
        else:
            mean_cos_turn = np.nan
            reversal_fraction = np.nan
        rows.append(
            dict(
                idx=i,
                n_steps=int(len(u)),
                resultant_length=res,
                mean_cos_turn=mean_cos_turn,
                reversal_fraction=reversal_fraction,
            )
        )
    return pd.DataFrame(rows)


def plot_comparison(
    obs_ac: pd.DataFrame,
    sim_ac: pd.DataFrame,
    obs_turn_deg: np.ndarray,
    sim_turn_deg: np.ndarray,
    obs_metrics: pd.DataFrame,
    sim_metrics: pd.DataFrame,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(obs_ac["lag"], obs_ac["mean"], marker="o", linewidth=2, color="#1d3557", label="Observed WT")
    ax.plot(sim_ac["lag"], sim_ac["mean"], marker="o", linewidth=2, color="#e76f51", label="Simulated model")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel(r"$\langle \hat{u}(t)\cdot\hat{u}(t+\tau)\rangle$")
    ax.set_title("Directional Autocorrelation")
    ax.legend(frameon=False)

    ax = axes[1]
    bins = np.linspace(0, 180, 37)
    ax.hist(obs_turn_deg, bins=bins, density=True, alpha=0.55, color="#1d3557", label="Observed WT")
    ax.hist(sim_turn_deg, bins=bins, density=True, alpha=0.55, color="#e76f51", label="Simulated model")
    ax.set_xlabel("Turning Angle (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Turning Angle Distribution")
    ax.legend(frameon=False)

    ax = axes[2]
    bins_r = np.linspace(0, 1, 31)
    ax.hist(
        obs_metrics["resultant_length"].to_numpy(dtype=float),
        bins=bins_r,
        density=True,
        alpha=0.55,
        color="#1d3557",
        label="Observed WT",
    )
    ax.hist(
        sim_metrics["resultant_length"].to_numpy(dtype=float),
        bins=bins_r,
        density=True,
        alpha=0.55,
        color="#e76f51",
        label="Simulated model",
    )
    ax.set_xlabel("Resultant Length")
    ax.set_ylabel("Density")
    ax.set_title("Track Directionality")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
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

    wt_pattern = re.compile(args.wt_regex, flags=re.IGNORECASE)
    obs_u_tracks, obs_steps_tracks, obs_turn_tracks = build_observed_tracks(
        df=df,
        source_col=args.source_col,
        wt_pattern=wt_pattern,
        class_filter=args.class_filter,
        min_track_len=args.min_track_len,
        step_eps=args.step_eps,
        direction_frame_lag=args.direction_frame_lag,
    )
    if not obs_u_tracks:
        raise RuntimeError("No WT tracks remained after filtering.")

    fit = fit_wt_process_moment_init(obs_turn_tracks, obs_steps_tracks)
    obs_lengths = np.asarray([len(u) for u in obs_u_tracks], dtype=int)

    n_sim_tracks = int(args.n_sim_tracks) if args.n_sim_tracks > 0 else int(len(obs_u_tracks))
    dir_fit, fit_diag = fit_direction_params_by_search(
        obs_u_tracks=obs_u_tracks,
        obs_turn_tracks=obs_turn_tracks,
        init_params=fit,
        max_lag=args.max_lag,
        n_sim_tracks=n_sim_tracks,
        n_iter=args.fit_search_iter,
        seed=args.fit_search_seed,
        score_resultant_weight=args.score_resultant_weight,
        score_reversal_weight=args.score_reversal_weight,
    )
    fit.update(dir_fit)
    fit.update(
        {
            "fit_objective_score": float(fit_diag["score"]),
            "fit_objective_ac_rmse": float(fit_diag["ac_rmse"]),
        }
    )

    rng = np.random.default_rng(args.seed)
    sim_u_tracks, sim_turn_tracks = simulate_direction_tracks(
        n_tracks=n_sim_tracks,
        lengths=obs_lengths,
        params=fit,
        rng=rng,
    )

    obs_ac = summarize_autocorr(obs_u_tracks, max_lag=args.max_lag)
    sim_ac = summarize_autocorr(sim_u_tracks, max_lag=args.max_lag)
    obs_metrics = track_metrics(obs_u_tracks, obs_turn_tracks)
    sim_metrics = track_metrics(sim_u_tracks, sim_turn_tracks)

    obs_turn_deg = np.degrees(np.abs(np.concatenate([t for t in obs_turn_tracks if len(t) > 0])))
    sim_turn_deg = np.degrees(np.abs(np.concatenate([t for t in sim_turn_tracks if len(t) > 0])))

    summary = {
        "fit_parameters": fit,
        "observed_wt": {
            "n_tracks": int(len(obs_u_tracks)),
            "mean_resultant_length": float(obs_metrics["resultant_length"].mean()),
            "mean_cos_turn": float(obs_metrics["mean_cos_turn"].mean(skipna=True)),
            "mean_reversal_fraction": float(obs_metrics["reversal_fraction"].mean(skipna=True)),
        },
        "simulated_model": {
            "n_tracks": int(len(sim_u_tracks)),
            "mean_resultant_length": float(sim_metrics["resultant_length"].mean()),
            "mean_cos_turn": float(sim_metrics["mean_cos_turn"].mean(skipna=True)),
            "mean_reversal_fraction": float(sim_metrics["reversal_fraction"].mean(skipna=True)),
        },
        "settings": {
            "source_col": args.source_col,
            "class_filter": args.class_filter,
            "wt_regex": args.wt_regex,
            "min_track_len": int(args.min_track_len),
            "direction_frame_lag": int(args.direction_frame_lag),
            "step_eps": float(args.step_eps),
            "max_lag": int(args.max_lag),
            "n_sim_tracks": int(n_sim_tracks),
            "fit_search_iter": int(args.fit_search_iter),
            "fit_search_seed": int(args.fit_search_seed),
            "score_resultant_weight": float(args.score_resultant_weight),
            "score_reversal_weight": float(args.score_reversal_weight),
            "seed": int(args.seed),
        },
        "fit_diagnostics": fit_diag,
    }

    params_json = args.output_dir / "wt_polarity_model_params.json"
    ac_cmp_csv = args.output_dir / "wt_polarity_model_autocorr_compare.csv"
    trk_cmp_csv = args.output_dir / "wt_polarity_model_track_metrics_compare.csv"
    fig_png = args.output_dir / "wt_polarity_model_comparison.png"

    with params_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ac_cmp_df = obs_ac.rename(columns={"mean": "obs_mean", "n_pairs": "obs_n_pairs", "std": "obs_std", "sem": "obs_sem"})
    ac_cmp_df = ac_cmp_df.merge(
        sim_ac.rename(columns={"mean": "sim_mean", "n_pairs": "sim_n_pairs", "std": "sim_std", "sem": "sim_sem"}),
        on="lag",
        how="outer",
        sort=True,
    )
    ac_cmp_df.to_csv(ac_cmp_csv, index=False)

    obs_metrics = obs_metrics.copy()
    obs_metrics["dataset"] = "observed_wt"
    sim_metrics = sim_metrics.copy()
    sim_metrics["dataset"] = "simulated_model"
    pd.concat([obs_metrics, sim_metrics], ignore_index=True).to_csv(trk_cmp_csv, index=False)

    plot_comparison(
        obs_ac=obs_ac,
        sim_ac=sim_ac,
        obs_turn_deg=obs_turn_deg,
        sim_turn_deg=sim_turn_deg,
        obs_metrics=obs_metrics,
        sim_metrics=sim_metrics,
        out_png=fig_png,
    )

    print(f"Observed WT tracks: {len(obs_u_tracks)}")
    print(f"Simulated tracks: {len(sim_u_tracks)}")
    print("Fitted parameters:")
    for k in ["p_rev", "kappa_fwd", "kappa_rev", "bias_mix", "step_log_mu", "step_log_sigma"]:
        print(f"  {k}: {fit[k]:.6f}")
    print(f"  fit_objective_score: {fit['fit_objective_score']:.6f}")
    print(f"  fit_objective_ac_rmse: {fit['fit_objective_ac_rmse']:.6f}")
    print(f"Saved: {params_json}")
    print(f"Saved: {ac_cmp_csv}")
    print(f"Saved: {trk_cmp_csv}")
    print(f"Saved: {fig_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
