#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
class TrackSeries:
    source: str
    track_id: int
    u: np.ndarray  # shape [T, 2]
    theta: np.ndarray  # shape [T]
    step: np.ndarray  # shape [T]


@dataclass
class PolarityFit:
    alpha_birth: float  # per-step Poisson rate
    tau_live: float  # steps
    p_birth: float  # per-step birth probability
    p_death: float  # per-step death probability
    rho_ou: float  # AR(1) coefficient in OU discretization
    tau_ou: float  # steps
    sigma_ou: float  # OU diffusion coefficient
    innovation_std: float  # residual std in AR(1)
    turn_event_thresh_deg: float
    n_events: int
    n_segments: int
    median_segment_len: float
    mean_inter_birth_steps: float
    mean_life_steps: float
    inter_birth_steps: list[int]
    life_steps: list[int]
    jump_angles: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit data-driven polarity dynamics (Poisson birth/death + OU orientation update) "
            "from tracked data, then couple the fitted polarity process to a mesoscopic-like "
            "biochemical-mechanical migration model."
        )
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/all_pv_tracked.csv"),
        help="Tracked CSV input file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/data_driven_polarity_mesoscopic"),
        help="Output directory.",
    )
    p.add_argument("--source-col", type=str, default=SOURCE_COL, help="Source column.")
    p.add_argument(
        "--class-filter",
        type=str,
        default="migrating dendritic cell",
        help="Case-insensitive class filter.",
    )
    p.add_argument(
        "--condition-regex",
        type=str,
        default=r"^wt(?:_|\d|$)",
        help="Regex applied to source filename stem for selecting tracks to fit.",
    )
    p.add_argument("--min-track-len", type=int, default=8, help="Minimum detections per track.")
    p.add_argument(
        "--direction-frame-lag",
        type=int,
        default=2,
        help="Frame lag used to compute direction vectors from positions.",
    )
    p.add_argument("--step-eps", type=float, default=5.0, help="Minimum step size (px) to keep.")
    p.add_argument(
        "--turn-event-thresh-deg",
        type=float,
        default=45.0,
        help="Turning-angle threshold (deg) for polarity event detection.",
    )
    p.add_argument("--max-lag", type=int, default=20, help="Max lag for autocorrelation comparison.")
    p.add_argument("--seed", type=int, default=17, help="Random seed.")

    # Simulation size
    p.add_argument(
        "--n-sims",
        type=int,
        default=120,
        help="Number of simulated trajectories in the coupled model.",
    )
    p.add_argument(
        "--sim-steps",
        type=int,
        default=0,
        help="Fixed steps per simulated trajectory; 0 samples from observed track-length distribution.",
    )

    # Coupled membrane/biochemistry model
    p.add_argument("--dt", type=float, default=1.0, help="Time step in model units.")
    p.add_argument("--n-membrane", type=int, default=96, help="Membrane nodes.")
    p.add_argument("--r0", type=float, default=1.0, help="Reference cell radius (model length units).")
    p.add_argument(
        "--source-kappa",
        type=float,
        default=16.0,
        help="Concentration of polarity source profile on membrane (von Mises style).",
    )
    p.add_argument(
        "--initial-active-sites",
        type=int,
        default=1,
        help="Number of initial active polarity sites at t=0.",
    )
    p.add_argument(
        "--max-active-sites",
        type=int,
        default=8,
        help="Maximum number of simultaneously active polarity sites.",
    )
    p.add_argument(
        "--lifetime-scale",
        type=float,
        default=1.25,
        help="Multiplier on empirical polarity-site lifetime to permit overlap between sites.",
    )
    p.add_argument(
        "--source-age-power",
        type=float,
        default=1.0,
        help="Exponent for age weighting of each active site contribution to source profile.",
    )
    p.add_argument(
        "--source-clip",
        type=float,
        default=2.0,
        help="Upper clip for summed multi-site source field; set <=0 to disable clipping.",
    )

    # Activator
    p.add_argument("--Da", type=float, default=0.18, help="Activator diffusion.")
    p.add_argument("--ra", type=float, default=0.30, help="Activator decay.")
    p.add_argument("--ba", type=float, default=2.5, help="Activator production.")
    p.add_argument("--amax", type=float, default=1.5, help="Activator saturation.")

    # Actin pools
    p.add_argument("--Df", type=float, default=0.05, help="F-actin diffusion.")
    p.add_argument("--Dg", type=float, default=0.08, help="G-actin diffusion.")
    p.add_argument("--pmax", type=float, default=1.8, help="Max F-actin polymerization.")
    p.add_argument("--mu-f", type=float, default=0.65, help="F-actin depolymerization.")
    p.add_argument("--Kg", type=float, default=0.6, help="G-actin half-saturation.")
    p.add_argument("--Kf", type=float, default=0.9, help="F-actin inhibition half-saturation.")
    p.add_argument("--hill-n", type=float, default=2.0, help="Hill exponent.")
    p.add_argument("--poly-base", type=float, default=0.10, help="Baseline polymerization drive.")
    p.add_argument("--poly-source-gain", type=float, default=0.90, help="Polarity-source gain on polymerization.")
    p.add_argument("--actin-total", type=float, default=1.8, help="Initial mean F+G actin total.")
    p.add_argument("--f0", type=float, default=0.65, help="Initial F-actin baseline.")

    # Myosin two-pool
    p.add_argument("--Dma", type=float, default=0.05, help="Membrane myosin diffusion.")
    p.add_argument("--Dmb", type=float, default=0.08, help="Cytosolic myosin diffusion.")
    p.add_argument("--Amax", type=float, default=0.95, help="Max soluble->membrane conversion.")
    p.add_argument("--mu-m", type=float, default=0.55, help="Membrane myosin release rate.")
    p.add_argument("--Kmb", type=float, default=0.55, help="Myosin half-saturation.")
    p.add_argument("--Kfm", type=float, default=0.80, help="F-actin suppression of membrane myosin.")
    p.add_argument("--myosin-total", type=float, default=1.2, help="Initial mean Ma+Mb myosin total.")
    p.add_argument("--ma0", type=float, default=0.50, help="Initial membrane myosin baseline.")

    # Mechanics
    p.add_argument("--k-r", type=float, default=0.40, help="Radial elastic stiffness to reference radius.")
    p.add_argument("--D-r", type=float, default=0.03, help="Radial smoothing diffusion on membrane.")
    p.add_argument("--beta-f", type=float, default=0.35, help="F-actin protrusive coefficient.")
    p.add_argument("--beta-m", type=float, default=0.50, help="Myosin contractile coefficient.")
    p.add_argument("--drag", type=float, default=45.0, help="Translational drag.")
    p.add_argument("--r-min", type=float, default=0.45, help="Minimum local radius clamp.")
    p.add_argument("--r-max", type=float, default=1.70, help="Maximum local radius clamp.")
    p.add_argument("--center-step-eps", type=float, default=1e-6, help="Minimum center step for direction stats.")

    return p.parse_args()


def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def circ_mean(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * angles))))


def periodic_laplacian(v: np.ndarray) -> np.ndarray:
    return np.roll(v, -1) + np.roll(v, 1) - 2.0 * v


def ensure_columns(df: pd.DataFrame, source_col: str) -> None:
    required = [source_col, "track_id", "frame_index", "x", "y", "class_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_track_directions(track_df: pd.DataFrame, step_eps: float, direction_frame_lag: int) -> tuple[np.ndarray, np.ndarray]:
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


def load_filtered_tracks(args: argparse.Namespace) -> list[TrackSeries]:
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
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
        raise RuntimeError("No rows left after class filter.")

    cond_pattern = re.compile(args.condition_regex, flags=re.IGNORECASE)
    tracks: list[TrackSeries] = []
    for (source, track_id), tr in df.groupby([args.source_col, "track_id"], sort=False):
        stem = Path(str(source)).stem
        if not cond_pattern.search(stem):
            continue
        if len(tr) < args.min_track_len:
            continue
        u, step = compute_track_directions(tr, step_eps=args.step_eps, direction_frame_lag=args.direction_frame_lag)
        if len(u) < max(2, args.min_track_len - 1):
            continue
        theta = np.arctan2(u[:, 1], u[:, 0])
        tracks.append(TrackSeries(source=str(source), track_id=int(track_id), u=u, theta=theta, step=step))
    if not tracks:
        raise RuntimeError("No tracks remained after condition and quality filters.")
    return tracks


def turn_event_indices(theta: np.ndarray, turn_event_thresh_rad: float) -> np.ndarray:
    n = len(theta)
    if n <= 1:
        return np.empty(0, dtype=int)
    dtheta = wrap_pi(theta[1:] - theta[:-1])
    return np.flatnonzero(np.abs(dtheta) >= turn_event_thresh_rad) + 1


def segment_track(theta: np.ndarray, turn_event_thresh_rad: float) -> list[tuple[int, int]]:
    n = len(theta)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    event_idx = turn_event_indices(theta, turn_event_thresh_rad=turn_event_thresh_rad)
    boundaries = np.concatenate(([0], event_idx, [n]))
    segments: list[tuple[int, int]] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s > 0:
            segments.append((int(s), int(e)))
    return segments


def fit_polarity_from_tracks(tracks: list[TrackSeries], dt: float, turn_event_thresh_deg: float) -> PolarityFit:
    turn_event_thresh_rad = math.radians(float(turn_event_thresh_deg))
    all_segment_lengths: list[int] = []
    all_jump_angles: list[float] = []
    all_y_t: list[float] = []
    all_y_tp1: list[float] = []
    total_steps = 0
    n_events = 0

    for tr in tracks:
        theta = tr.theta
        total_steps += len(theta)
        segs = segment_track(theta, turn_event_thresh_rad=turn_event_thresh_rad)
        if not segs:
            continue
        all_segment_lengths.extend([e - s for s, e in segs])
        n_events += max(0, len(segs) - 1)

        target_angles: list[float] = []
        for s, e in segs:
            targ = circ_mean(theta[s:e])
            target_angles.append(targ)
            if e - s >= 2:
                y = wrap_pi(theta[s:e] - targ)
                all_y_t.extend(y[:-1].tolist())
                all_y_tp1.extend(y[1:].tolist())
        if len(target_angles) >= 2:
            for a0, a1 in zip(target_angles[:-1], target_angles[1:]):
                all_jump_angles.append(float(wrap_pi(a1 - a0)))

    if total_steps <= 0:
        raise RuntimeError("No direction steps available for polarity fitting.")

    alpha_birth = float(max(1e-9, n_events / total_steps))
    tau_live = float(np.mean(all_segment_lengths)) if all_segment_lengths else 1.0
    tau_live = max(1e-6, tau_live)
    p_birth = 1.0 - math.exp(-alpha_birth * dt)
    p_death = 1.0 - math.exp(-dt / tau_live)

    if len(all_y_t) >= 8:
        y_t = np.asarray(all_y_t, dtype=float)
        y_tp1 = np.asarray(all_y_tp1, dtype=float)
        denom = float(np.sum(y_t * y_t))
        rho = float(np.sum(y_t * y_tp1) / denom) if denom > 1e-12 else 0.7
    else:
        rho = 0.7
    rho = float(np.clip(rho, 1e-4, 0.9995))
    tau_ou = float(-dt / math.log(rho))

    if len(all_y_t) >= 8:
        y_t = np.asarray(all_y_t, dtype=float)
        y_tp1 = np.asarray(all_y_tp1, dtype=float)
        residual = y_tp1 - rho * y_t
        innovation_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else float(np.std(residual))
    else:
        innovation_std = 0.25
    innovation_std = max(1e-6, innovation_std)
    sigma_ou = float(innovation_std / (tau_ou * math.sqrt(max(1e-12, 1.0 - rho**2))))

    return PolarityFit(
        alpha_birth=alpha_birth,
        tau_live=tau_live,
        p_birth=p_birth,
        p_death=p_death,
        rho_ou=rho,
        tau_ou=tau_ou,
        sigma_ou=sigma_ou,
        innovation_std=innovation_std,
        turn_event_thresh_deg=float(turn_event_thresh_deg),
        n_events=int(n_events),
        n_segments=int(len(all_segment_lengths)),
        median_segment_len=float(np.median(all_segment_lengths)) if all_segment_lengths else 0.0,
        jump_angles=[float(x) for x in all_jump_angles],
    )


def summarize_autocorr_from_u(u_tracks: list[np.ndarray], max_lag: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for lag in range(1, max_lag + 1):
        vals: list[float] = []
        for u in u_tracks:
            if len(u) <= lag:
                continue
            vals.extend(np.sum(u[:-lag] * u[lag:], axis=1).tolist())
        arr = np.asarray(vals, dtype=float)
        n = len(arr)
        rows.append(
            dict(
                lag=lag,
                n_pairs=int(n),
                mean=float(np.mean(arr)) if n else np.nan,
                std=float(np.std(arr, ddof=1)) if n > 1 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def track_metrics(u_tracks: list[np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for i, u in enumerate(u_tracks, start=1):
        if len(u) == 0:
            continue
        res = float(np.linalg.norm(np.mean(u, axis=0)))
        if len(u) >= 2:
            dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
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


def vm_source_profile(node_angles: np.ndarray, theta: float, kappa: float) -> np.ndarray:
    # smooth source around current polarity direction
    raw = np.exp(kappa * (np.cos(node_angles - theta) - 1.0))
    mx = float(np.max(raw))
    if mx <= 0:
        return np.zeros_like(raw)
    return raw / mx


def simulate_coupled_model(
    fit: PolarityFit,
    track_lengths: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    n_sims = int(max(1, args.n_sims))
    node_n = int(max(12, args.n_membrane))
    node_angles = np.linspace(0.0, 2.0 * np.pi, node_n, endpoint=False)
    nvec = np.stack([np.cos(node_angles), np.sin(node_angles)], axis=1)
    dt = float(args.dt)

    # OU innovation scale in discretized form (angle update)
    ou_step_std = float(fit.sigma_ou * fit.tau_ou * math.sqrt(max(1e-12, 1.0 - fit.rho_ou**2)))

    sim_rows: list[dict[str, float]] = []
    profile_rows: list[dict[str, float]] = []
    sim_u_tracks: list[np.ndarray] = []

    jump_pool = np.asarray(fit.jump_angles, dtype=float) if fit.jump_angles else np.array([], dtype=float)
    lengths = track_lengths if len(track_lengths) > 0 else np.array([120], dtype=int)

    for sim_id in range(1, n_sims + 1):
        steps = int(args.sim_steps) if int(args.sim_steps) > 0 else int(rng.choice(lengths))
        steps = max(20, steps)

        center = np.zeros(2, dtype=float)
        center_prev = center.copy()
        center_u: list[np.ndarray] = []

        # polarity state
        active = False
        theta = float(rng.uniform(-np.pi, np.pi))
        rprot = theta

        # membrane and biochemical fields
        r = np.full(node_n, float(args.r0), dtype=float)
        a = np.zeros(node_n, dtype=float)
        F = np.full(node_n, float(args.f0), dtype=float)
        G = np.full(node_n, max(1e-6, float(args.actin_total) - float(args.f0)), dtype=float)
        Ma = np.full(node_n, float(args.ma0), dtype=float)
        Mb = np.full(node_n, max(1e-6, float(args.myosin_total) - float(args.ma0)), dtype=float)

        for t in range(steps):
            birth = False
            death = False

            if rng.random() < fit.p_birth:
                birth = True
                active = True
                if jump_pool.size > 0:
                    jump = float(rng.choice(jump_pool))
                    rprot = float(wrap_pi(rprot + jump))
                else:
                    rprot = float(rng.uniform(-np.pi, np.pi))

            if active and (not birth) and (rng.random() < fit.p_death):
                death = True
                active = False

            if active:
                theta = float(wrap_pi(rprot + fit.rho_ou * wrap_pi(theta - rprot) + ou_step_std * rng.normal()))
            else:
                theta = float(wrap_pi(theta + 0.5 * ou_step_std * rng.normal()))

            source = vm_source_profile(node_angles=node_angles, theta=theta, kappa=float(args.source_kappa)) if active else np.zeros(node_n, dtype=float)

            lap_a = periodic_laplacian(a)
            a = a + dt * (float(args.Da) * lap_a - float(args.ra) * a + float(args.ba) * source * (float(args.amax) - a))
            a = np.clip(a, 0.0, float(args.amax))

            # actin dynamics: localized polymerization by polarity source and activator
            lap_F = periodic_laplacian(F)
            lap_G = periodic_laplacian(G)
            n_h = float(args.hill_n)
            g_term = (G**n_h) / (G**n_h + float(args.Kg) ** n_h + 1e-9)
            f_term = (float(args.Kf) ** n_h) / (float(args.Kf) ** n_h + F**n_h + 1e-9)
            source_drive = float(args.poly_base) + float(args.poly_source_gain) * source * (a / (float(args.amax) + 1e-9))
            polymer = float(args.pmax) * g_term * f_term * source_drive
            dF = float(args.Df) * lap_F + polymer - float(args.mu_f) * F
            dG = float(args.Dg) * lap_G - polymer + float(args.mu_f) * F
            F = np.clip(F + dt * dF, 0.0, None)
            G = np.clip(G + dt * dG, 1e-9, None)

            # conserve mean actin approximately
            actin_mean = np.mean(F + G)
            if actin_mean > 1e-9:
                scale = float(args.actin_total) / actin_mean
                F *= scale
                G *= scale

            # myosin two-pool with F-dependent suppression
            lap_Ma = periodic_laplacian(Ma)
            lap_Mb = periodic_laplacian(Mb)
            mb_term = (Mb**n_h) / (Mb**n_h + float(args.Kmb) ** n_h + 1e-9)
            fm_term = (float(args.Kfm) ** n_h) / (float(args.Kfm) ** n_h + F**n_h + 1e-9)
            conv = float(args.Amax) * mb_term * fm_term
            dMa = float(args.Dma) * lap_Ma + conv - float(args.mu_m) * Ma
            dMb = float(args.Dmb) * lap_Mb - conv + float(args.mu_m) * Ma
            Ma = np.clip(Ma + dt * dMa, 0.0, None)
            Mb = np.clip(Mb + dt * dMb, 1e-9, None)

            # conserve mean myosin approximately
            myo_mean = np.mean(Ma + Mb)
            if myo_mean > 1e-9:
                scale_m = float(args.myosin_total) / myo_mean
                Ma *= scale_m
                Mb *= scale_m

            # mechanics: radial deformation + center motion from active stresses
            active_scalar = float(args.beta_f) * F - float(args.beta_m) * Ma
            lap_r = periodic_laplacian(r)
            dr = float(args.D_r) * lap_r + float(args.k_r) * (float(args.r0) - r) + active_scalar
            r = np.clip(r + dt * dr, float(args.r_min), float(args.r_max))

            net_force = np.sum(active_scalar[:, None] * nvec, axis=0)
            velocity = net_force / float(args.drag)
            center = center + dt * velocity

            dcenter = center - center_prev
            speed = float(np.linalg.norm(dcenter))
            if speed > float(args.center_step_eps):
                center_u.append(dcenter / speed)
            center_prev = center.copy()

            sim_rows.append(
                dict(
                    sim_id=sim_id,
                    step=t,
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    theta=float(theta),
                    rprot=float(rprot),
                    active=int(active),
                    birth=int(birth),
                    death=int(death),
                    speed=speed,
                    mean_activator=float(np.mean(a)),
                    mean_F=float(np.mean(F)),
                    mean_G=float(np.mean(G)),
                    mean_Ma=float(np.mean(Ma)),
                    mean_Mb=float(np.mean(Mb)),
                    mean_radius=float(np.mean(r)),
                    radius_cv=float(np.std(r) / (np.mean(r) + 1e-12)),
                )
            )

            # keep profile output lightweight
            if (t % 5) == 0:
                source_idx = int(np.argmax(source)) if active else -1
                profile_rows.append(
                    dict(
                        sim_id=sim_id,
                        step=t,
                        source_idx=source_idx,
                        source_peak=float(np.max(source)) if source.size else 0.0,
                        F_peak=float(np.max(F)),
                        Ma_peak=float(np.max(Ma)),
                        radius_peak=float(np.max(r)),
                        radius_min=float(np.min(r)),
                    )
                )

        if center_u:
            sim_u_tracks.append(np.stack(center_u, axis=0))

    return pd.DataFrame(sim_rows), pd.DataFrame(profile_rows), sim_u_tracks


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_filtered_tracks(args)
    fit = fit_polarity_from_tracks(tracks, dt=float(args.dt), turn_event_thresh_deg=float(args.turn_event_thresh_deg))

    obs_u_tracks = [t.u for t in tracks]
    obs_track_lengths = np.asarray([len(t.u) for t in tracks], dtype=int)
    obs_ac = summarize_autocorr_from_u(obs_u_tracks, max_lag=int(args.max_lag))
    obs_metrics = track_metrics(obs_u_tracks)

    sim_df, profile_df, sim_u_tracks = simulate_coupled_model(fit=fit, track_lengths=obs_track_lengths, args=args)
    sim_ac = summarize_autocorr_from_u(sim_u_tracks, max_lag=int(args.max_lag))
    sim_metrics = track_metrics(sim_u_tracks)

    ac_cmp = obs_ac.rename(columns={"n_pairs": "obs_n_pairs", "mean": "obs_mean", "std": "obs_std"}).merge(
        sim_ac.rename(columns={"n_pairs": "sim_n_pairs", "mean": "sim_mean", "std": "sim_std"}),
        on="lag",
        how="outer",
        sort=True,
    )
    if len(ac_cmp) > 0:
        d = ac_cmp[["obs_mean", "sim_mean"]].dropna()
        ac_rmse = float(np.sqrt(np.mean((d["obs_mean"].to_numpy() - d["sim_mean"].to_numpy()) ** 2))) if len(d) else float("nan")
    else:
        ac_rmse = float("nan")

    summary = {
        "fit": {
            "alpha_birth_per_step": fit.alpha_birth,
            "tau_live_steps": fit.tau_live,
            "p_birth_per_step": fit.p_birth,
            "p_death_per_step": fit.p_death,
            "rho_ou": fit.rho_ou,
            "tau_ou_steps": fit.tau_ou,
            "sigma_ou": fit.sigma_ou,
            "innovation_std": fit.innovation_std,
            "turn_event_thresh_deg": fit.turn_event_thresh_deg,
            "n_events": fit.n_events,
            "n_segments": fit.n_segments,
            "median_segment_len": fit.median_segment_len,
            "n_jump_samples": len(fit.jump_angles),
        },
        "observed": {
            "n_tracks": int(len(obs_u_tracks)),
            "mean_resultant_length": float(obs_metrics["resultant_length"].mean()),
            "mean_cos_turn": float(obs_metrics["mean_cos_turn"].mean(skipna=True)),
            "mean_reversal_fraction": float(obs_metrics["reversal_fraction"].mean(skipna=True)),
        },
        "simulated_coupled_model": {
            "n_trajectories": int(len(sim_u_tracks)),
            "mean_resultant_length": float(sim_metrics["resultant_length"].mean()) if len(sim_metrics) else float("nan"),
            "mean_cos_turn": float(sim_metrics["mean_cos_turn"].mean(skipna=True)) if len(sim_metrics) else float("nan"),
            "mean_reversal_fraction": float(sim_metrics["reversal_fraction"].mean(skipna=True)) if len(sim_metrics) else float("nan"),
            "autocorr_rmse": ac_rmse,
        },
        "settings": {
            "input_csv": str(args.input_csv),
            "source_col": args.source_col,
            "class_filter": args.class_filter,
            "condition_regex": args.condition_regex,
            "min_track_len": int(args.min_track_len),
            "direction_frame_lag": int(args.direction_frame_lag),
            "step_eps": float(args.step_eps),
            "dt": float(args.dt),
            "n_sims": int(args.n_sims),
            "sim_steps": int(args.sim_steps),
            "seed": int(args.seed),
        },
    }

    fit_json = args.output_dir / "data_driven_polarity_fit.json"
    sim_csv = args.output_dir / "coupled_mesoscopic_simulation_timeseries.csv"
    profile_csv = args.output_dir / "coupled_mesoscopic_profile_summary.csv"
    ac_csv = args.output_dir / "coupled_mesoscopic_autocorr_compare.csv"
    metrics_csv = args.output_dir / "coupled_mesoscopic_track_metrics_compare.csv"
    fig_png = args.output_dir / "coupled_mesoscopic_polarity_compare.png"

    with fit_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sim_df.to_csv(sim_csv, index=False)
    profile_df.to_csv(profile_csv, index=False)
    ac_cmp.to_csv(ac_csv, index=False)
    obs_metrics = obs_metrics.copy()
    obs_metrics["dataset"] = "observed_data"
    sim_metrics = sim_metrics.copy()
    sim_metrics["dataset"] = "simulated_coupled_model"
    pd.concat([obs_metrics, sim_metrics], ignore_index=True).to_csv(metrics_csv, index=False)

    # figure: autocorrelation + track directionality + example center trajectories
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    if len(obs_ac):
        ax.plot(obs_ac["lag"], obs_ac["mean"], marker="o", color="#1d3557", linewidth=2, label="Observed")
    if len(sim_ac):
        ax.plot(sim_ac["lag"], sim_ac["mean"], marker="o", color="#e76f51", linewidth=2, label="Coupled model")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel(r"$\langle \hat{u}(t)\cdot\hat{u}(t+\tau)\rangle$")
    ax.set_title("Directional Autocorrelation")
    ax.legend(frameon=False)

    ax = axes[1]
    bins_r = np.linspace(0, 1, 31)
    if len(obs_metrics):
        ax.hist(
            obs_metrics["resultant_length"].to_numpy(dtype=float),
            bins=bins_r,
            density=True,
            alpha=0.6,
            color="#1d3557",
            label="Observed",
        )
    if len(sim_metrics):
        ax.hist(
            sim_metrics["resultant_length"].to_numpy(dtype=float),
            bins=bins_r,
            density=True,
            alpha=0.6,
            color="#e76f51",
            label="Coupled model",
        )
    ax.set_xlabel("Resultant Length")
    ax.set_ylabel("Density")
    ax.set_title("Track Directionality")
    ax.legend(frameon=False)

    ax = axes[2]
    show_sims = sim_df["sim_id"].drop_duplicates().head(12).tolist() if len(sim_df) else []
    for sid in show_sims:
        tr = sim_df[sim_df["sim_id"] == sid]
        ax.plot(tr["center_x"], tr["center_y"], linewidth=1.0, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (model units)")
    ax.set_ylabel("y (model units)")
    ax.set_title("Example Simulated Trajectories")

    fig.tight_layout()
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    print(f"Tracks used for fit: {len(tracks)}")
    print("Fitted polarity parameters:")
    print(f"  alpha_birth_per_step: {fit.alpha_birth:.6f}")
    print(f"  tau_live_steps: {fit.tau_live:.6f}")
    print(f"  p_birth_per_step: {fit.p_birth:.6f}")
    print(f"  p_death_per_step: {fit.p_death:.6f}")
    print(f"  rho_ou: {fit.rho_ou:.6f}")
    print(f"  tau_ou_steps: {fit.tau_ou:.6f}")
    print(f"  sigma_ou: {fit.sigma_ou:.6f}")
    print(f"Autocorrelation RMSE (observed vs coupled model): {ac_rmse:.6f}")
    print(f"Saved: {fit_json}")
    print(f"Saved: {sim_csv}")
    print(f"Saved: {profile_csv}")
    print(f"Saved: {ac_csv}")
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {fig_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
