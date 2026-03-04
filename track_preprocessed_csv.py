#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


@dataclass
class TrackState:
    track_id: int
    last_frame: int
    cx: float
    cy: float
    area: float
    row_indices: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Track detections in preprocessed CSV using linear assignment (Hungarian algorithm), "
            "grouped by source video and class."
        )
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/all_pv_preprocessed.csv"),
        help="Input preprocessed CSV.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration/all_pv_tracked.csv"),
        help="Aggregated tracked CSV output path.",
    )
    p.add_argument(
        "--source-col",
        type=str,
        default=SOURCE_COL,
        help="Column name that identifies the source video path.",
    )
    p.add_argument(
        "--per-source",
        action="store_true",
        help="Write per-source tracked CSV files next to each source video.",
    )
    p.add_argument(
        "--per-source-suffix",
        type=str,
        default="_tracked",
        help="Suffix for per-source tracked CSV filename.",
    )

    # Assignment/gating
    p.add_argument("--max-link-dist-px", type=float, default=120.0, help="Max center distance for linking.")
    p.add_argument("--max-link-gap", type=int, default=1, help="Max number of missing frames allowed in link.")
    p.add_argument(
        "--max-area-ratio",
        type=float,
        default=4.0,
        help="Max area ratio between linked detections (symmetric; e.g. 4 means [1/4, 4]).",
    )
    p.add_argument(
        "--area-cost-weight",
        type=float,
        default=15.0,
        help="Weight for area change term in assignment cost.",
    )
    p.add_argument(
        "--max-assignment-cost",
        type=float,
        default=9999.0,
        help="Upper bound to accept assignment after gating.",
    )

    # Track post-filter
    p.add_argument("--min-track-len", type=int, default=1, help="Minimum track length.")
    p.add_argument("--drop-short-tracks", action="store_true", help="Drop rows belonging to short tracks.")
    return p.parse_args()


def ensure_columns(df: pd.DataFrame, source_col: str) -> None:
    required = ["frame_index", "x", "y", "width", "height", "class_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if source_col not in df.columns:
        # Fallback: treat all rows as one source inferred from input filename later.
        df[source_col] = ""


def _calc_area(w: float, h: float) -> float:
    return max(1e-6, float(w) * float(h))


def _assignment_cost(
    det_row: pd.Series,
    tr: TrackState,
    max_link_dist_px: float,
    max_area_ratio: float,
    area_cost_weight: float,
) -> float:
    cx = float(det_row["x"]) + float(det_row["width"]) / 2.0
    cy = float(det_row["y"]) + float(det_row["height"]) / 2.0
    area = _calc_area(float(det_row["width"]), float(det_row["height"]))
    dist = math.hypot(cx - tr.cx, cy - tr.cy)
    if dist > max_link_dist_px:
        return math.inf

    ratio = area / max(1e-6, tr.area)
    if ratio > max_area_ratio or ratio < 1.0 / max_area_ratio:
        return math.inf

    area_term = abs(math.log(ratio))
    return dist + area_cost_weight * area_term


def _infer_role_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    class_name = df["class_name"].astype(str).str.lower()
    is_pv = class_name.str.contains(r"vacuole|\bpv\b|parasite", regex=True, na=False)
    is_cell = (~is_pv) & class_name.str.contains(r"dendritic|cell", regex=True, na=False)
    if "class_id" in df.columns:
        cid = pd.to_numeric(df["class_id"], errors="coerce")
        is_pv = is_pv | (cid == 1)
        is_cell = (is_cell | (cid == 0)) & (~is_pv)
    return is_cell, is_pv


def annotate_cell_pv_size(src_df: pd.DataFrame) -> pd.DataFrame:
    out = src_df.copy()
    out["pv_size"] = 0.0
    if len(out) == 0:
        return out

    is_cell, is_pv = _infer_role_masks(out)
    if not bool(is_cell.any()) or not bool(is_pv.any()):
        return out

    by_frame = out.groupby("frame_index", sort=False)
    for _, frame_df in by_frame:
        cell_idx = frame_df.index[is_cell.loc[frame_df.index]].to_numpy()
        pv_idx = frame_df.index[is_pv.loc[frame_df.index]].to_numpy()
        if len(cell_idx) == 0 or len(pv_idx) == 0:
            continue

        cell_x = out.loc[cell_idx, "x"].to_numpy(dtype=float)
        cell_y = out.loc[cell_idx, "y"].to_numpy(dtype=float)
        cell_w = out.loc[cell_idx, "width"].to_numpy(dtype=float)
        cell_h = out.loc[cell_idx, "height"].to_numpy(dtype=float)
        cell_x2 = cell_x + cell_w
        cell_y2 = cell_y + cell_h
        cell_cx = cell_x + cell_w / 2.0
        cell_cy = cell_y + cell_h / 2.0

        pv_x = out.loc[pv_idx, "x"].to_numpy(dtype=float)
        pv_y = out.loc[pv_idx, "y"].to_numpy(dtype=float)
        pv_w = out.loc[pv_idx, "width"].to_numpy(dtype=float)
        pv_h = out.loc[pv_idx, "height"].to_numpy(dtype=float)
        pv_cx = pv_x + pv_w / 2.0
        pv_cy = pv_y + pv_h / 2.0
        pv_area = np.maximum(0.0, pv_w) * np.maximum(0.0, pv_h)

        for p_i, p_row_idx in enumerate(pv_idx):
            contain = (
                (pv_cx[p_i] >= cell_x)
                & (pv_cx[p_i] <= cell_x2)
                & (pv_cy[p_i] >= cell_y)
                & (pv_cy[p_i] <= cell_y2)
            )
            if not contain.any():
                continue
            cand = np.flatnonzero(contain)
            if len(cand) == 1:
                chosen_local = int(cand[0])
            else:
                dist = np.hypot(cell_cx[cand] - pv_cx[p_i], cell_cy[cand] - pv_cy[p_i])
                chosen_local = int(cand[int(np.argmin(dist))])
            chosen_idx = int(cell_idx[chosen_local])
            out.at[chosen_idx, "pv_size"] = float(out.at[chosen_idx, "pv_size"]) + float(pv_area[p_i])

    return out


def track_one_source(
    src_df: pd.DataFrame,
    max_link_dist_px: float,
    max_link_gap: int,
    max_area_ratio: float,
    area_cost_weight: float,
    max_assignment_cost: float,
    min_track_len: int,
    drop_short_tracks: bool,
) -> pd.DataFrame:
    src_df = src_df.copy()
    src_df["track_id"] = -1
    src_df["track_len"] = 0
    src_df["track_class_id"] = -1

    next_track_id = 0

    # Separate tracks per class key (class_id preferred when available, else class_name).
    if "class_id" in src_df.columns:
        class_key = src_df["class_id"].astype(str).where(src_df["class_id"].notna(), src_df["class_name"].astype(str))
    else:
        class_key = src_df["class_name"].astype(str)
    src_df["_class_track_key"] = class_key

    for _, class_part in src_df.groupby("_class_track_key", sort=False):
        # Per-class, per-frame assignment.
        class_indices = class_part.index.to_numpy()
        active: list[TrackState] = []
        completed: list[TrackState] = []
        class_track_counter = 0

        by_frame = class_part.groupby("frame_index", sort=True)
        for frame_value, frame_df in by_frame:
            frame = int(frame_value)
            det_indices = frame_df.index.to_numpy()

            # Expire stale tracks.
            kept_active: list[TrackState] = []
            for tr in active:
                if frame - tr.last_frame > max_link_gap + 1:
                    completed.append(tr)
                else:
                    kept_active.append(tr)
            active = kept_active

            # Assignment matrix.
            if active and len(det_indices) > 0:
                cost = np.full((len(active), len(det_indices)), fill_value=np.inf, dtype=float)
                for i, tr in enumerate(active):
                    for j, idx in enumerate(det_indices):
                        c = _assignment_cost(
                            src_df.loc[idx],
                            tr,
                            max_link_dist_px=max_link_dist_px,
                            max_area_ratio=max_area_ratio,
                            area_cost_weight=area_cost_weight,
                        )
                        cost[i, j] = c

                finite = np.isfinite(cost)
                if finite.any():
                    work = np.where(finite, cost, 1e12)
                    row_ind, col_ind = linear_sum_assignment(work)
                else:
                    row_ind = np.array([], dtype=int)
                    col_ind = np.array([], dtype=int)
            else:
                row_ind = np.array([], dtype=int)
                col_ind = np.array([], dtype=int)
                cost = None

            assigned_track_idx: set[int] = set()
            assigned_det_idx: set[int] = set()
            for ri, ci in zip(row_ind, col_ind):
                c = float(cost[ri, ci]) if cost is not None else math.inf
                if not math.isfinite(c) or c > max_assignment_cost:
                    continue
                tr = active[ri]
                idx = int(det_indices[ci])
                cx = float(src_df.at[idx, "x"]) + float(src_df.at[idx, "width"]) / 2.0
                cy = float(src_df.at[idx, "y"]) + float(src_df.at[idx, "height"]) / 2.0
                area = _calc_area(float(src_df.at[idx, "width"]), float(src_df.at[idx, "height"]))

                tr.last_frame = frame
                tr.cx = cx
                tr.cy = cy
                tr.area = area
                tr.row_indices.append(idx)
                assigned_track_idx.add(ri)
                assigned_det_idx.add(ci)

            # New tracks for unassigned detections.
            for local_j, idx in enumerate(det_indices):
                if local_j in assigned_det_idx:
                    continue
                cx = float(src_df.at[idx, "x"]) + float(src_df.at[idx, "width"]) / 2.0
                cy = float(src_df.at[idx, "y"]) + float(src_df.at[idx, "height"]) / 2.0
                area = _calc_area(float(src_df.at[idx, "width"]), float(src_df.at[idx, "height"]))
                tr = TrackState(
                    track_id=next_track_id,
                    last_frame=frame,
                    cx=cx,
                    cy=cy,
                    area=area,
                    row_indices=[int(idx)],
                )
                # stash per-class local counter in a dynamic attribute
                tr.class_track_id = class_track_counter
                class_track_counter += 1
                next_track_id += 1
                active.append(tr)

        completed.extend(active)

        # Write IDs/lengths back.
        for tr in completed:
            tlen = len(tr.row_indices)
            if tlen < min_track_len and drop_short_tracks:
                continue
            for idx in tr.row_indices:
                src_df.at[idx, "track_id"] = int(tr.track_id)
                src_df.at[idx, "track_len"] = int(tlen)
                src_df.at[idx, "track_class_id"] = int(getattr(tr, "class_track_id", -1))

    if drop_short_tracks:
        src_df = src_df[src_df["track_id"] >= 0].copy()

    src_df.drop(columns=["_class_track_key"], inplace=True)
    src_df.sort_values(by=["frame_index", "class_name", "track_id"], inplace=True)
    return src_df


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    ensure_columns(df, args.source_col)

    # If no source paths are present, infer one from input csv stem.
    if df[args.source_col].astype(str).str.strip().eq("").all():
        inferred = str(args.input_csv.with_suffix(".mp4").resolve())
        df[args.source_col] = inferred

    # Normalize numeric fields.
    for col in ["frame_index", "x", "y", "width", "height"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame_index", "x", "y", "width", "height"]).copy()
    df["frame_index"] = df["frame_index"].astype(int)

    out_groups: list[pd.DataFrame] = []
    total_sources = df[args.source_col].nunique()
    global_track_offset = 0

    for i, (source_path, src_df) in enumerate(df.groupby(args.source_col, sort=False), start=1):
        print(f"[{i}/{total_sources}] Tracking source: {source_path}")
        tracked = track_one_source(
            src_df,
            max_link_dist_px=args.max_link_dist_px,
            max_link_gap=args.max_link_gap,
            max_area_ratio=args.max_area_ratio,
            area_cost_weight=args.area_cost_weight,
            max_assignment_cost=args.max_assignment_cost,
            min_track_len=args.min_track_len,
            drop_short_tracks=args.drop_short_tracks,
        )
        tracked = annotate_cell_pv_size(tracked)
        tracked["track_id_local"] = tracked["track_id"]
        valid_local = tracked["track_id_local"] >= 0
        if valid_local.any():
            tracked.loc[valid_local, "track_id"] = tracked.loc[valid_local, "track_id_local"] + global_track_offset
            global_track_offset += int(tracked.loc[valid_local, "track_id_local"].nunique())

        out_groups.append(tracked)

        if args.per_source:
            source_p = Path(str(source_path))
            out_path = source_p.with_name(f"{source_p.stem}{args.per_source_suffix}.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tracked.to_csv(out_path, index=False)

    out_df = pd.concat(out_groups, ignore_index=True) if out_groups else df.iloc[0:0].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    n_tracks = int(out_df["track_id"].nunique()) if "track_id" in out_df.columns else 0
    print(f"Saved tracked CSV: {args.output_csv}")
    print(f"Rows: {len(out_df)}")
    print(f"Tracks: {n_tracks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
