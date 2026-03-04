#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


@dataclass
class Detection:
    row: dict[str, str]
    source_csv: Path
    frame: int
    obj_id: int
    class_id: int | None
    class_name: str
    class_key: str
    role: str
    x: float
    y: float
    w: float
    h: float
    score: float
    fov_w: float
    fov_h: float
    keep: bool = True
    drop_reason: str = ""

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class TrackState:
    class_key: str
    dets: list[Detection] = field(default_factory=list)
    last_frame: int = -1
    cx: float = 0.0
    cy: float = 0.0

    def add(self, det: Detection) -> None:
        self.dets.append(det)
        self.last_frame = det.frame
        self.cx = det.cx
        self.cy = det.cy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess segmentation CSVs for tracking: edge exclusion, in-frame duplicate suppression, "
            "PV-cell spatial constraint, temporal persistence filtering, and optional drift correction."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/Files/Dendritic_Cells/Migration"),
        help="Root folder containing segmentation CSV files.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_pv.csv",
        help="CSV glob pattern under root.",
    )
    p.add_argument(
        "--per-file-suffix",
        type=str,
        default="_preprocessed",
        help="Suffix for per-file output CSV.",
    )
    p.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Path for aggregated preprocessed CSV. Default: <root>/all_pv_preprocessed.csv",
    )
    p.add_argument(
        "--drop-log",
        type=Path,
        default=None,
        help="Path for dropped-row log CSV. Default: <root>/all_pv_preprocess_drops.csv",
    )

    # Step 1: edge exclusion
    p.add_argument("--edge-margin-px", type=float, default=-1.0, help="Boundary margin in px. -1 uses ratio.")
    p.add_argument(
        "--edge-margin-ratio",
        type=float,
        default=0.01,
        help="Boundary margin ratio of min(FOV width, height) when --edge-margin-px is -1.",
    )

    # Optional confidence filtering
    p.add_argument("--enable-conf-filter", action="store_true", help="Enable confidence threshold filtering.")
    p.add_argument("--min-score", type=float, default=0.0, help="Global minimum segment score.")
    p.add_argument("--cell-min-score", type=float, default=None, help="Optional cell-specific min score.")
    p.add_argument("--pv-min-score", type=float, default=None, help="Optional PV-specific min score.")

    # Step 4: in-frame duplicate suppression
    p.add_argument("--dup-iou-thresh", type=float, default=0.6, help="IoU threshold for duplicate suppression.")
    p.add_argument(
        "--dup-containment-thresh",
        type=float,
        default=0.85,
        help="Containment threshold for duplicate suppression.",
    )
    p.add_argument(
        "--dup-center-dist-px",
        type=float,
        default=40.0,
        help="Center distance threshold (px) for containment-based duplicate suppression.",
    )

    # Step 8: class constraint
    p.add_argument(
        "--cell-keywords",
        type=str,
        default="migrating dendritic cell,dendritic cell,cell",
        help="Comma-separated keywords for cell class_name matching.",
    )
    p.add_argument(
        "--pv-keywords",
        type=str,
        default="parasite vacuole,pv,vacuole",
        help="Comma-separated keywords for parasite vacuole class_name matching.",
    )
    p.add_argument(
        "--cell-class-ids",
        type=str,
        default="",
        help="Comma-separated class IDs treated as cells.",
    )
    p.add_argument(
        "--pv-class-ids",
        type=str,
        default="",
        help="Comma-separated class IDs treated as parasite vacuoles.",
    )
    p.add_argument(
        "--pv-cell-margin-px",
        type=float,
        default=15.0,
        help="Cell bbox expansion margin (px) when checking PV center containment.",
    )
    p.add_argument(
        "--no-pv-require-cell",
        action="store_true",
        help="Do not drop PV detections on frames without any cell detection.",
    )

    # Step 5: temporal persistence
    p.add_argument("--min-track-len", type=int, default=3, help="Minimum proto-track length to keep.")
    p.add_argument("--link-max-dist-px", type=float, default=120.0, help="Max linking distance between frames.")
    p.add_argument("--max-link-gap", type=int, default=1, help="Max frame gap allowed for linking.")

    # Step 9: optional drift correction
    p.add_argument("--enable-drift-correction", action="store_true", help="Enable global drift correction.")
    p.add_argument(
        "--drift-max-match-dist-px",
        type=float,
        default=120.0,
        help="Max nearest-neighbor distance used for drift estimation.",
    )
    return p.parse_args()


def parse_keywords(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def parse_id_set(text: str) -> set[int]:
    out: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def float_str(v: float) -> str:
    s = f"{v:.6f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def infer_role(
    class_id: int | None,
    class_name: str,
    cell_keywords: list[str],
    pv_keywords: list[str],
    cell_class_ids: set[int],
    pv_class_ids: set[int],
) -> str:
    if class_id is not None:
        if class_id in cell_class_ids:
            return "cell"
        if class_id in pv_class_ids:
            return "pv"
    n = class_name.lower()
    if any(k in n for k in pv_keywords):
        return "pv"
    if any(k in n for k in cell_keywords):
        return "cell"
    return "other"


def load_detections(
    csv_path: Path,
    cell_keywords: list[str],
    pv_keywords: list[str],
    cell_class_ids: set[int],
    pv_class_ids: set[int],
) -> tuple[list[str], list[Detection]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if not header:
            return [], []

        dets: list[Detection] = []
        inferred_fov_w = 0.0
        inferred_fov_h = 0.0

        for row in reader:
            if not any((v or "").strip() for v in row.values()):
                continue

            frame = parse_int(row.get("frame_index", "0"))
            obj_id = parse_int(row.get("obj_id", "0"))
            x = parse_float(row.get("x", "0"))
            y = parse_float(row.get("y", "0"))
            w = parse_float(row.get("width", "0"))
            h = parse_float(row.get("height", "0"))
            score = parse_float(row.get("segment_score", "0"))
            fov_w = parse_float(row.get("fov_width", "0"))
            fov_h = parse_float(row.get("fov_height", "0"))

            if fov_w <= 0:
                inferred_fov_w = max(inferred_fov_w, x + w)
            if fov_h <= 0:
                inferred_fov_h = max(inferred_fov_h, y + h)

            class_id = parse_int(row.get("class_id", "")) if "class_id" in row else None
            class_name = (row.get("class_name", "") or "").strip()
            class_key = f"id:{class_id}" if class_id is not None else f"name:{class_name.lower() or 'unknown'}"
            role = infer_role(class_id, class_name, cell_keywords, pv_keywords, cell_class_ids, pv_class_ids)

            dets.append(
                Detection(
                    row=row,
                    source_csv=csv_path,
                    frame=frame,
                    obj_id=obj_id,
                    class_id=class_id,
                    class_name=class_name,
                    class_key=class_key,
                    role=role,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    score=score,
                    fov_w=fov_w,
                    fov_h=fov_h,
                )
            )

    if inferred_fov_w > 0 or inferred_fov_h > 0:
        for d in dets:
            if d.fov_w <= 0:
                d.fov_w = inferred_fov_w
            if d.fov_h <= 0:
                d.fov_h = inferred_fov_h

    return header, dets


def raw_video_path_from_csv(csv_path: Path) -> str:
    return str(csv_path.with_suffix(".mp4").resolve())


def box_iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def box_containment(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    denom = min(max(a.area, 1e-9), max(b.area, 1e-9))
    return inter / denom


def center_distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def mark_drop(det: Detection, reason: str) -> None:
    if det.keep:
        det.keep = False
        det.drop_reason = reason


def edge_filter(dets: Iterable[Detection], edge_margin_px: float, edge_margin_ratio: float) -> None:
    for d in dets:
        margin = edge_margin_px if edge_margin_px >= 0 else edge_margin_ratio * min(d.fov_w, d.fov_h)
        if d.x1 <= margin or d.y1 <= margin or d.x2 >= d.fov_w - margin or d.y2 >= d.fov_h - margin:
            mark_drop(d, "edge_hit")


def confidence_filter(
    dets: Iterable[Detection],
    enable: bool,
    min_score: float,
    cell_min_score: float | None,
    pv_min_score: float | None,
) -> None:
    if not enable:
        return
    for d in dets:
        threshold = min_score
        if d.role == "cell" and cell_min_score is not None:
            threshold = cell_min_score
        elif d.role == "pv" and pv_min_score is not None:
            threshold = pv_min_score
        if d.score < threshold:
            mark_drop(d, "low_score")


def estimate_drift(
    dets: list[Detection],
    drift_max_match_dist_px: float,
) -> dict[int, tuple[float, float]]:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in dets:
        by_frame[d.frame].append(d)
    frames = sorted(by_frame)
    if not frames:
        return {}

    # Prefer cell detections for drift; fallback to all detections when no cells exist.
    has_cells = any(d.role == "cell" for d in dets)
    centers: dict[int, np.ndarray] = {}
    for f in frames:
        rows = [d for d in by_frame[f] if (d.role == "cell" if has_cells else True)]
        if rows:
            centers[f] = np.array([[d.cx, d.cy] for d in rows], dtype=np.float64)
        else:
            centers[f] = np.empty((0, 2), dtype=np.float64)

    drift: dict[int, tuple[float, float]] = {}
    cum_dx = 0.0
    cum_dy = 0.0
    drift[frames[0]] = (0.0, 0.0)

    for prev_f, curr_f in zip(frames[:-1], frames[1:]):
        prev = centers[prev_f]
        curr = centers[curr_f]
        if len(prev) == 0 or len(curr) == 0:
            step_dx = 0.0
            step_dy = 0.0
        else:
            deltas: list[tuple[float, float]] = []
            for c in curr:
                diffs = prev - c[None, :]
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                j = int(np.argmin(dists))
                if float(dists[j]) <= drift_max_match_dist_px:
                    deltas.append((float(c[0] - prev[j, 0]), float(c[1] - prev[j, 1])))
            if deltas:
                step_dx = float(np.median([d[0] for d in deltas]))
                step_dy = float(np.median([d[1] for d in deltas]))
            else:
                step_dx = 0.0
                step_dy = 0.0

        cum_dx += step_dx
        cum_dy += step_dy
        drift[curr_f] = (cum_dx, cum_dy)

    return drift


def apply_drift_correction(
    dets: list[Detection],
    drift_max_match_dist_px: float,
) -> None:
    drift = estimate_drift(dets, drift_max_match_dist_px)
    for d in dets:
        dx, dy = drift.get(d.frame, (0.0, 0.0))
        d.x -= dx
        d.y -= dy


def in_frame_duplicate_suppression(
    dets: list[Detection],
    dup_iou_thresh: float,
    dup_containment_thresh: float,
    dup_center_dist_px: float,
) -> None:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in dets:
        by_frame[d.frame].append(d)

    prev_kept_by_class: dict[str, list[Detection]] = defaultdict(list)

    for frame in sorted(by_frame):
        frame_rows = [d for d in by_frame[frame] if d.keep]
        by_class: dict[str, list[Detection]] = defaultdict(list)
        for d in frame_rows:
            by_class[d.class_key].append(d)

        current_kept_by_class: dict[str, list[Detection]] = defaultdict(list)
        for class_key, class_rows in by_class.items():
            prev_rows = prev_kept_by_class.get(class_key, [])

            scored: list[tuple[float, Detection]] = []
            for d in class_rows:
                best_iou_prev = 0.0
                area_penalty = 0.0
                if prev_rows:
                    best_prev = None
                    for p in prev_rows:
                        iou = box_iou(d, p)
                        if iou > best_iou_prev:
                            best_iou_prev = iou
                            best_prev = p
                    if best_prev is not None and best_prev.area > 0 and d.area > 0:
                        area_penalty = abs(math.log(d.area / best_prev.area))

                priority = d.score + 0.5 * best_iou_prev - 0.15 * area_penalty
                scored.append((priority, d))

            scored.sort(key=lambda x: x[0], reverse=True)
            kept: list[Detection] = []
            for _, cand in scored:
                is_dup = False
                for k in kept:
                    iou = box_iou(cand, k)
                    contain = box_containment(cand, k)
                    dist = center_distance(cand, k)
                    if iou >= dup_iou_thresh or (contain >= dup_containment_thresh and dist <= dup_center_dist_px):
                        is_dup = True
                        break
                if is_dup:
                    mark_drop(cand, "duplicate_in_frame")
                else:
                    kept.append(cand)

            current_kept_by_class[class_key] = kept

        prev_kept_by_class = current_kept_by_class


def class_constraint_filter(
    dets: list[Detection],
    pv_cell_margin_px: float,
    pv_require_cell: bool,
) -> None:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in dets:
        if d.keep:
            by_frame[d.frame].append(d)

    for frame in sorted(by_frame):
        rows = by_frame[frame]
        cells = [d for d in rows if d.role == "cell"]
        pvs = [d for d in rows if d.role == "pv"]
        if not pvs:
            continue

        if not cells:
            if pv_require_cell:
                for pv in pvs:
                    mark_drop(pv, "pv_without_cell_frame")
            continue

        expanded = [
            (
                c.x1 - pv_cell_margin_px,
                c.y1 - pv_cell_margin_px,
                c.x2 + pv_cell_margin_px,
                c.y2 + pv_cell_margin_px,
            )
            for c in cells
        ]
        for pv in pvs:
            inside = any(x1 <= pv.cx <= x2 and y1 <= pv.cy <= y2 for x1, y1, x2, y2 in expanded)
            if not inside:
                mark_drop(pv, "pv_outside_cell_context")


def temporal_persistence_filter(
    dets: list[Detection],
    min_track_len: int,
    link_max_dist_px: float,
    max_link_gap: int,
) -> None:
    if min_track_len <= 1:
        return

    by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in dets:
        if d.keep:
            by_class[d.class_key].append(d)

    for class_key, class_dets in by_class.items():
        frame_map: dict[int, list[Detection]] = defaultdict(list)
        for d in class_dets:
            frame_map[d.frame].append(d)
        frames = sorted(frame_map)
        active_tracks: list[TrackState] = []
        finished_tracks: list[TrackState] = []

        for frame in frames:
            current_dets = frame_map[frame]

            still_active: list[TrackState] = []
            for tr in active_tracks:
                if frame - tr.last_frame > max_link_gap:
                    finished_tracks.append(tr)
                else:
                    still_active.append(tr)
            active_tracks = still_active

            pairs: list[tuple[float, int, int]] = []
            for di, d in enumerate(current_dets):
                for ti, tr in enumerate(active_tracks):
                    dist = math.hypot(d.cx - tr.cx, d.cy - tr.cy)
                    if dist <= link_max_dist_px:
                        pairs.append((dist, di, ti))
            pairs.sort(key=lambda x: x[0])

            used_dets: set[int] = set()
            used_tracks: set[int] = set()
            for _, di, ti in pairs:
                if di in used_dets or ti in used_tracks:
                    continue
                active_tracks[ti].add(current_dets[di])
                used_dets.add(di)
                used_tracks.add(ti)

            for di, d in enumerate(current_dets):
                if di in used_dets:
                    continue
                tr = TrackState(class_key=class_key)
                tr.add(d)
                active_tracks.append(tr)

        finished_tracks.extend(active_tracks)

        for tr in finished_tracks:
            if len(tr.dets) < min_track_len:
                for d in tr.dets:
                    mark_drop(d, "short_track")


def render_row(det: Detection, header: list[str]) -> dict[str, str]:
    out = {k: det.row.get(k, "") for k in header}
    # Keep schema stable while writing potentially drift-corrected positions.
    if "x" in out:
        out["x"] = float_str(det.x)
    if "y" in out:
        out["y"] = float_str(det.y)
    if "width" in out:
        out["width"] = float_str(det.w)
    if "height" in out:
        out["height"] = float_str(det.h)
    if "segment_score" in out:
        out["segment_score"] = float_str(det.score)
    if "fov_width" in out:
        out["fov_width"] = float_str(det.fov_w)
    if "fov_height" in out:
        out["fov_height"] = float_str(det.fov_h)
    return out


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def preprocess_one_file(
    csv_path: Path,
    args: argparse.Namespace,
    cell_keywords: list[str],
    pv_keywords: list[str],
    cell_class_ids: set[int],
    pv_class_ids: set[int],
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]], dict[str, int], Path]:
    header, dets = load_detections(csv_path, cell_keywords, pv_keywords, cell_class_ids, pv_class_ids)
    if not header:
        return [], [], [], {}, csv_path.with_name(f"{csv_path.stem}{args.per_file_suffix}.csv")

    edge_filter(dets, args.edge_margin_px, args.edge_margin_ratio)
    confidence_filter(dets, args.enable_conf_filter, args.min_score, args.cell_min_score, args.pv_min_score)

    active = [d for d in dets if d.keep]
    if args.enable_drift_correction and active:
        apply_drift_correction(active, args.drift_max_match_dist_px)

    in_frame_duplicate_suppression(
        active,
        dup_iou_thresh=args.dup_iou_thresh,
        dup_containment_thresh=args.dup_containment_thresh,
        dup_center_dist_px=args.dup_center_dist_px,
    )
    class_constraint_filter(active, args.pv_cell_margin_px, pv_require_cell=not args.no_pv_require_cell)
    temporal_persistence_filter(
        active,
        min_track_len=args.min_track_len,
        link_max_dist_px=args.link_max_dist_px,
        max_link_gap=args.max_link_gap,
    )

    kept = [d for d in dets if d.keep]
    dropped = [d for d in dets if not d.keep]

    kept.sort(key=lambda d: (d.frame, d.class_key, d.obj_id))
    per_file_rows = [render_row(d, header) for d in kept]

    source_video = raw_video_path_from_csv(csv_path)
    drop_rows = [
        {
            "source_csv": str(csv_path),
            SOURCE_COL: source_video,
            "frame_index": str(d.frame),
            "obj_id": str(d.obj_id),
            "class_id": "" if d.class_id is None else str(d.class_id),
            "class_name": d.class_name,
            "drop_reason": d.drop_reason,
            "x": float_str(d.x),
            "y": float_str(d.y),
            "width": float_str(d.w),
            "height": float_str(d.h),
            "segment_score": float_str(d.score),
        }
        for d in dropped
    ]

    reason_counts: dict[str, int] = defaultdict(int)
    for d in dropped:
        reason_counts[d.drop_reason] += 1

    out_path = csv_path.with_name(f"{csv_path.stem}{args.per_file_suffix}.csv")
    return header, per_file_rows, drop_rows, dict(reason_counts), out_path


def main() -> int:
    args = parse_args()
    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    files = sorted(p for p in root.rglob(args.pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found under {root} with pattern '{args.pattern}'")

    cell_keywords = parse_keywords(args.cell_keywords)
    pv_keywords = parse_keywords(args.pv_keywords)
    cell_class_ids = parse_id_set(args.cell_class_ids)
    pv_class_ids = parse_id_set(args.pv_class_ids)

    aggregate_output = args.aggregate_output or (root / "all_pv_preprocessed.csv")
    drop_log_path = args.drop_log or (root / "all_pv_preprocess_drops.csv")

    aggregate_header: list[str] | None = None
    aggregate_rows: list[dict[str, str]] = []
    all_drop_rows: list[dict[str, str]] = []
    total_in = 0
    total_out = 0
    reason_totals: dict[str, int] = defaultdict(int)

    for i, csv_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Preprocessing {csv_path}")
        header, per_file_rows, drop_rows, reason_counts, out_path = preprocess_one_file(
            csv_path, args, cell_keywords, pv_keywords, cell_class_ids, pv_class_ids
        )
        total_in += len(per_file_rows) + len(drop_rows)
        total_out += len(per_file_rows)
        for k, v in reason_counts.items():
            reason_totals[k] += v

        write_csv(out_path, header, per_file_rows)

        if aggregate_header is None:
            aggregate_header = list(header)
            if SOURCE_COL not in aggregate_header:
                aggregate_header.append(SOURCE_COL)

        source_video = raw_video_path_from_csv(csv_path)
        for row in per_file_rows:
            row2 = dict(row)
            row2[SOURCE_COL] = source_video
            aggregate_rows.append(row2)

        all_drop_rows.extend(drop_rows)

    if aggregate_header is None:
        raise RuntimeError("No CSV header found while processing files.")
    write_csv(aggregate_output, aggregate_header, aggregate_rows)

    if all_drop_rows:
        drop_header = [
            "source_csv",
            SOURCE_COL,
            "frame_index",
            "obj_id",
            "class_id",
            "class_name",
            "drop_reason",
            "x",
            "y",
            "width",
            "height",
            "segment_score",
        ]
        write_csv(drop_log_path, drop_header, all_drop_rows)

    print(f"Preprocessed files: {len(files)}")
    print(f"Input rows: {total_in}")
    print(f"Kept rows: {total_out}")
    print(f"Dropped rows: {total_in - total_out}")
    print(f"Aggregate output: {aggregate_output}")
    print(f"Drop log: {drop_log_path}")
    if reason_totals:
        print("Drop reasons:")
        for reason, count in sorted(reason_totals.items(), key=lambda x: x[0]):
            print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
