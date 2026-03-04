#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics.models.sam import SAM3VideoSemanticPredictor

DEFAULT_VIDEO_ROOT = Path("/mnt/Files/Dendritic_Cells/Migration")
DEFAULT_MODEL_PATH = Path("/home/jianfei/SegmentAnything/sam3.pt")
DEFAULT_PROMPT = "migrating dendritic cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 text-prompt video segmentation on all MP4 files under a root directory. "
            "For each video, save <video_stem>.csv and <video_stem>_pred.avi next to the input video."
        )
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Parent directory that contains MP4 files in subfolders.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to sam3.pt model.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        type=str,
        default=[],
        help="Text prompt used for segmentation. Repeat --prompt to use multiple prompts.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=644, help="Inference image size.")
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="_pred",
        help="Suffix used for saved prediction video names.",
    )
    parser.add_argument(
        "--pred-ext",
        type=str,
        default=".avi",
        help="Prediction video extension (default .avi). Use .mp4 only if your system has the right codec.",
    )
    return parser.parse_args()


def build_predictor(model_path: Path, device: str, conf: float, imgsz: int) -> SAM3VideoSemanticPredictor:
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        half=False,
        device=device,
        imgsz=imgsz,
    )
    return SAM3VideoSemanticPredictor(overrides=overrides)


def discover_mp4_files(root: Path, pred_suffix: str) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.mp4")
        if p.is_file() and not p.name.endswith(f"{pred_suffix}.mp4")
    )


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 3.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 3.0


def normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ".avi"
    return ext if ext.startswith(".") else f".{ext}"


def build_video_writer(path: Path, fps: float, frame_size: tuple[int, int], ext: str) -> cv2.VideoWriter:
    # Prefer AVI codecs for portability; MP4 codecs vary by system decoder availability.
    if ext == ".avi":
        codec_candidates = ["MJPG", "XVID", "DIVX"]
    elif ext == ".mp4":
        codec_candidates = ["avc1", "mp4v"]
    else:
        codec_candidates = ["XVID", "MJPG", "mp4v"]

    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        f"Failed to create video writer for {path} with codecs {codec_candidates}. "
        "Try --pred-ext .avi on this machine."
    )


def class_name_from_result(names: dict | list | tuple | None, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def process_video(
    video_path: Path,
    predictor: SAM3VideoSemanticPredictor,
    prompts: list[str],
    pred_suffix: str,
    pred_ext: str,
) -> tuple[Path, Path]:
    csv_path = video_path.with_suffix(".csv")
    pred_video_path = video_path.with_name(f"{video_path.stem}{pred_suffix}{pred_ext}")
    fps = get_video_fps(video_path)

    predictor.inference_state = {}
    results = predictor(source=str(video_path), text=prompts, stream=True)

    video_writer = None
    fov_width = None
    fov_height = None
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_index",
                    "obj_id",
                    "class_id",
                    "class_name",
                    "x",
                    "y",
                    "width",
                    "height",
                    "segment_score",
                    "fov_width",
                    "fov_height",
                ]
            )

            for frame_idx, result in enumerate(results):
                if fov_width is None or fov_height is None:
                    img_h, img_w = result.orig_img.shape[:2]
                    fov_width = int(img_w)
                    fov_height = int(img_h)

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
                    track_ids = boxes.id
                    if track_ids is None:
                        obj_ids = list(range(len(xyxy)))
                    else:
                        obj_ids = track_ids.cpu().numpy().astype(int).tolist()

                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = box.tolist()
                        writer.writerow(
                            [
                                frame_idx,
                                int(obj_ids[i]),
                                int(cls_ids[i]),
                                class_name_from_result(result.names, int(cls_ids[i])),
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                                float(scores[i]),
                                fov_width,
                                fov_height,
                            ]
                        )

                plotted = result.plot()
                if video_writer is None:
                    h, w = plotted.shape[:2]
                    video_writer = build_video_writer(pred_video_path, fps, (w, h), pred_ext)
                video_writer.write(plotted)
    finally:
        if video_writer is not None:
            video_writer.release()

    return csv_path, pred_video_path


def main() -> int:
    args = parse_args()
    pred_ext = normalize_ext(args.pred_ext)
    prompts = [p.strip() for p in args.prompt if p and p.strip()]
    if not prompts:
        prompts = [DEFAULT_PROMPT]
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.video_root.exists():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")

    videos = discover_mp4_files(args.video_root, args.pred_suffix)
    if not videos:
        print(f"No MP4 files found under: {args.video_root}")
        return 0

    predictor = build_predictor(args.model, args.device, args.conf, args.imgsz)

    failures: list[Path] = []
    for i, video_path in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] Processing: {video_path}")
        try:
            csv_path, pred_path = process_video(video_path, predictor, prompts, args.pred_suffix, pred_ext)
            print(f"  saved CSV: {csv_path}")
            print(f"  saved predictions: {pred_path}")
        except Exception as exc:
            failures.append(video_path)
            print(f"  failed: {video_path}")
            print(f"  reason: {exc}")

    if failures:
        print(f"Completed with {len(failures)} failure(s).")
        for p in failures:
            print(f"  - {p}")
        return 1

    print(f"Completed successfully. Processed {len(videos)} video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
