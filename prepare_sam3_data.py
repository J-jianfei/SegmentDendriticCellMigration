#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import tifffile as tiff

# Order to assign colors when labels omit a color.
DEFAULT_LABEL_COLORS = ["green", "red", "blue"]
COLOR_TO_CHANNEL = {"red": 0, "green": 1, "blue": 2}


@dataclass
class LabelSpec:
    name: str
    index: int
    color: str
    image: np.ndarray | None = None


@dataclass
class ChannelBCOverride:
    target_type: str  # "brightfield", "label", or "index"
    target_value: str | int
    brightness: float
    contrast: float


def parse_label_spec(spec: str) -> Tuple[str, int, str | None]:
    """
    Parse label spec in the form NAME:INDEX or NAME:INDEX:COLOR.
    """
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid --label '{spec}'. Use NAME:INDEX or NAME:INDEX:COLOR")
    name = parts[0].strip()
    if not name:
        raise ValueError(f"Invalid --label '{spec}': empty name")
    try:
        index = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid --label '{spec}': INDEX must be int")
    color = parts[2].strip().lower() if len(parts) == 3 else None
    if color and color not in COLOR_TO_CHANNEL:
        raise ValueError(f"Invalid --label '{spec}': COLOR must be red/green/blue")
    return name, index, color


def parse_channel_bc_spec(spec: str) -> Tuple[str, float, float]:
    """
    Parse per-channel B/C spec in the form TARGET:BRIGHTNESS:CONTRAST.
    TARGET can be 'brightfield', label name, or channel index.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --bc-channel '{spec}'. "
            "Use TARGET:BRIGHTNESS:CONTRAST (e.g. brightfield:60:45 or pv:80:90 or 3:40:70)."
        )
    target = parts[0].strip()
    if not target:
        raise ValueError(f"Invalid --bc-channel '{spec}': empty TARGET")
    try:
        brightness = float(parts[1])
        contrast = float(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid --bc-channel '{spec}': BRIGHTNESS and CONTRAST must be numeric percentages"
        ) from exc
    return target, brightness, contrast


def load_tiff_with_axes(path: Path) -> Tuple[np.ndarray, str | None]:
    """Load TIFF and return array plus axes metadata if available."""
    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        axes = getattr(series, "axes", None)
        arr = series.asarray()
    return arr, axes


def reorder_to_thwc(arr: np.ndarray, axes: str | None) -> np.ndarray:
    """
    Reorder array to T x H x W x C.
    Uses axes metadata when available; falls back to heuristics otherwise.
    """
    if axes:
        axes = axes.upper()

        # Drop Z by taking the first index if present.
        if "Z" in axes:
            z_idx = axes.index("Z")
            arr = np.take(arr, 0, axis=z_idx)
            axes = axes.replace("Z", "")

        # Add missing dims.
        if "T" not in axes:
            arr = arr[None, ...]
            axes = "T" + axes
        if "C" not in axes:
            arr = arr[..., None]
            axes = axes + "C"

        # Build permutation to T, Y, X, C.
        perm = [axes.index(a) for a in "TYXC"]
        return np.transpose(arr, perm)

    # Heuristics when axes metadata is missing.
    if arr.ndim == 2:
        return arr[None, ..., None]
    if arr.ndim == 3:
        # Assume HWC if last dim is channels; else CHW; else HWT.
        if arr.shape[-1] in {1, 2, 3, 4}:
            return arr[None, ...]
        if arr.shape[0] in {1, 2, 3, 4}:
            return np.transpose(arr, (1, 2, 0))[None, ...]
        # Treat as HWT (no channels)
        return np.transpose(arr, (2, 0, 1))[..., None]
    if arr.ndim == 4:
        # Assume last dim is time.
        core = arr
        # Find channel dim among first three dims.
        c_dim = None
        for i in range(3):
            if core.shape[i] in {1, 2, 3, 4}:
                c_dim = i
                break
        if c_dim is None:
            # Treat as HWT with no channels.
            return np.transpose(arr, (3, 0, 1, 2))[..., None]
        # Move to H,W,C,T then to T,H,W,C.
        if c_dim == 0:
            hwct = np.transpose(arr, (1, 2, 0, 3))
        elif c_dim == 1:
            hwct = np.transpose(arr, (0, 2, 1, 3))
        else:
            hwct = arr
        return np.transpose(hwct, (3, 0, 1, 2))

    raise ValueError(f"Unsupported TIFF shape {arr.shape}; please provide axes metadata")


def compute_channel_stats(
    frames: np.ndarray,
    method: str,
    p_low: float,
    p_high: float,
    stat_stride: int,
    max_samples: int,
) -> List[Tuple[float, float]]:
    """Compute per-channel (vmin, vmax) from sampled frames."""
    if method == "none":
        return []

    _, _, _, c = frames.shape
    stats: List[Tuple[float, float]] = []
    for ch in range(c):
        sample = frames[::stat_stride, :, :, ch].reshape(-1)
        stats.append(compute_stats_for_sample(sample, method, p_low, p_high, max_samples))
    return stats


def compute_stats_for_sample(
    sample: np.ndarray, method: str, p_low: float, p_high: float, max_samples: int
) -> Tuple[float, float]:
    if sample.size > max_samples:
        idx = np.random.choice(sample.size, size=max_samples, replace=False)
        sample = sample[idx]
    if method == "minmax":
        vmin, vmax = float(sample.min()), float(sample.max())
    else:
        vmin, vmax = np.percentile(sample, [p_low, p_high]).astype(np.float32)
    return float(vmin), float(vmax)


def resolve_percentiles(method: str, p_low: float, p_high: float, imagej_sat: float) -> Tuple[float, float]:
    if method == "imagej":
        sat = max(0.0, min(imagej_sat, 100.0))
        half = sat / 2.0
        return half, 100.0 - half
    return p_low, p_high


def compute_frame_stats(
    frame: np.ndarray, channels: List[int], method: str, p_low: float, p_high: float, max_samples: int
) -> List[Tuple[float, float]]:
    """Compute per-channel (vmin, vmax) for a single frame."""
    stats: List[Tuple[float, float]] = []
    for ch in channels:
        sample = frame[..., ch].reshape(-1)
        stats.append(compute_stats_for_sample(sample, method, p_low, p_high, max_samples))
    return stats


def scale_to_uint8(
    img: np.ndarray,
    method: str,
    stats: Tuple[float, float] | None,
    bc_brightness: float,
    bc_contrast: float,
) -> np.ndarray:
    """Scale a single-channel image to uint8, then apply ImageJ-like brightness/contrast."""
    if method == "none":
        base = img.astype(np.float32)
        base = np.clip(base, 0, 255) / 255.0
        adjusted = apply_imagej_bc(base, bc_brightness, bc_contrast)
        return (adjusted * 255.0).astype(np.uint8)

    if stats is None:
        raise ValueError("Stats required for normalization")
    vmin, vmax = stats
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)

    base = img.astype(np.float32)
    base = (base - vmin) / (vmax - vmin)
    base = np.clip(base, 0, 1)
    adjusted = apply_imagej_bc(base, bc_brightness, bc_contrast)
    return (adjusted * 255.0).astype(np.uint8)


def build_rgb(
    brightfield: np.ndarray,
    labels: List[LabelSpec],
    normalize: str,
    bf_stats: Tuple[float, float] | None,
    label_stats: List[Tuple[float, float]] | None,
    bf_bc: Tuple[float, float],
    label_bcs: List[Tuple[float, float]],
) -> np.ndarray:
    """Combine brightfield and label channels into RGB."""
    h, w = brightfield.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    bf_brightness, bf_contrast = bf_bc
    bf_u8 = scale_to_uint8(brightfield, normalize, bf_stats, bf_brightness, bf_contrast)
    rgb[..., 0] = bf_u8
    rgb[..., 1] = bf_u8
    rgb[..., 2] = bf_u8

    if labels and label_stats is None:
        raise ValueError("Label stats required")
    if len(label_bcs) != len(labels):
        raise ValueError("Label B/C overrides must align with labels")

    for i, spec in enumerate(labels):
        if spec.image is None:
            raise ValueError(f"Label '{spec.name}' has no image data")
        ch = COLOR_TO_CHANNEL[spec.color]
        lbl_brightness, lbl_contrast = label_bcs[i]
        lbl_u8 = scale_to_uint8(spec.image, normalize, label_stats[i], lbl_brightness, lbl_contrast)
        rgb[..., ch] = np.clip(
            rgb[..., ch].astype(np.uint16) + lbl_u8.astype(np.uint16), 0, 255
        ).astype(np.uint8)

    return rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare multi-channel TIFFs for SAM3 by composing brightfield + labels into MP4 videos.")
    p.add_argument("--input", required=True, type=Path, help="Input TIFF file or directory")
    p.add_argument("--output", required=True, type=Path, help="Output directory for MP4s")
    p.add_argument("--brightfield", required=True, type=int, help="Channel index for brightfield")
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label spec NAME:INDEX or NAME:INDEX:COLOR (color = red/green/blue). Can be repeated.",
    )
    p.add_argument(
        "--normalize",
        choices=["percentile", "minmax", "none", "imagej"],
        default="percentile",
        help="Normalization for each channel before uint8 conversion",
    )
    p.add_argument(
        "--imagej-sat",
        type=float,
        default=0.35,
        help="ImageJ auto-contrast saturated pixels percent (default 0.35)",
    )
    p.add_argument(
        "--normalize-scope",
        choices=["global", "frame"],
        default="global",
        help="Use global stats across all frames or adaptive per-frame stats",
    )
    p.add_argument("--p-low", type=float, default=1.0, help="Low percentile for normalization")
    p.add_argument("--p-high", type=float, default=99.0, help="High percentile for normalization")
    p.add_argument("--fps", type=float, default=3.0, help="FPS for output video")
    p.add_argument("--stat-stride", type=int, default=1, help="Frame stride for stats sampling")
    p.add_argument("--max-samples", type=int, default=1_000_000, help="Max samples per channel for stats")
    p.add_argument(
        "--bc-brightness",
        type=float,
        default=50.0,
        help=(
            "ImageJ-style brightness percentage (0..100). "
            "50 is neutral, 100 shifts the transfer function fully to the left."
        ),
    )
    p.add_argument(
        "--bc-contrast",
        type=float,
        default=50.0,
        help=(
            "ImageJ-style contrast percentage (0..100). "
            "50 is neutral, 100 approaches infinite slope."
        ),
    )
    p.add_argument(
        "--bc-channel",
        action="append",
        default=[],
        help=(
            "Per-channel B/C override: TARGET:BRIGHTNESS:CONTRAST. "
            "TARGET can be brightfield, a label name, or a channel index. "
            "Can be repeated; last matching override wins."
        ),
    )
    p.add_argument(
        "--output-structure",
        choices=["mirror", "flat"],
        default="mirror",
        help="Mirror input directory structure or write all outputs to a flat directory",
    )
    return p.parse_args()


def iter_tiff_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.tif")) + sorted(path.rglob("*.tiff"))


def clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def imagej_window_from_percent(
    default_min: float, default_max: float, brightness_pct: float, contrast_pct: float
) -> Tuple[float, float]:
    """Compute ImageJ-like display window from brightness/contrast percentages."""
    value_range = default_max - default_min
    if value_range <= 0:
        return default_min, default_max

    b = clamp_percent(brightness_pct)
    c = clamp_percent(contrast_pct)

    # ImageJ brightness convention: larger value shifts window center to lower intensities (left).
    center = default_min + value_range * ((100.0 - b) / 100.0)

    # ImageJ contrast convention around a midpoint: 50 is slope 1, 100 approaches inf, 0 approaches 0.
    if c < 50.0:
        slope = c / 50.0
    elif c > 50.0:
        denom = 100.0 - c
        slope = float("inf") if denom <= 0 else 50.0 / denom
    else:
        slope = 1.0

    if slope == 0.0:
        half_width = value_range * 1e6
    elif np.isinf(slope):
        half_width = 0.0
    else:
        half_width = 0.5 * value_range / slope

    return center - half_width, center + half_width


def apply_imagej_bc(norm: np.ndarray, brightness_pct: float, contrast_pct: float) -> np.ndarray:
    """Apply ImageJ-like brightness/contrast on normalized [0,1] data."""
    win_min, win_max = imagej_window_from_percent(0.0, 1.0, brightness_pct, contrast_pct)
    width = win_max - win_min
    if width <= 1e-12:
        # Infinite slope: threshold at the window center.
        center = 0.5 * (win_min + win_max)
        return (norm >= center).astype(np.float32)
    out = (norm - win_min) / width
    return np.clip(out, 0.0, 1.0)


def build_channel_bc_overrides(specs: List[str], labels: List[LabelSpec]) -> List[ChannelBCOverride]:
    label_name_keys = {}
    for lbl in labels:
        key = lbl.name.strip().lower()
        if key in label_name_keys:
            raise ValueError(
                f"Duplicate label name '{lbl.name}' in --label list. "
                "Use unique names when targeting --bc-channel by label name."
            )
        label_name_keys[key] = lbl.index

    overrides: List[ChannelBCOverride] = []
    for spec in specs:
        target, brightness, contrast = parse_channel_bc_spec(spec)
        target_key = target.strip().lower()
        if target_key == "brightfield":
            overrides.append(ChannelBCOverride("brightfield", "brightfield", brightness, contrast))
            continue
        if target_key in label_name_keys:
            overrides.append(ChannelBCOverride("label", target_key, brightness, contrast))
            continue
        try:
            ch_index = int(target)
            overrides.append(ChannelBCOverride("index", ch_index, brightness, contrast))
            continue
        except ValueError as exc:
            valid_labels = ", ".join(sorted(label_name_keys.keys())) if label_name_keys else "none"
            raise ValueError(
                f"Invalid --bc-channel target '{target}'. "
                f"Use brightfield, one of label names [{valid_labels}], or an integer channel index."
            ) from exc
    return overrides


def resolve_channel_bc(
    channel_index: int,
    is_brightfield: bool,
    label_name: str | None,
    default_brightness: float,
    default_contrast: float,
    overrides: List[ChannelBCOverride],
) -> Tuple[float, float]:
    brightness = default_brightness
    contrast = default_contrast
    label_key = label_name.strip().lower() if label_name else None

    for item in overrides:
        if item.target_type == "brightfield" and is_brightfield:
            brightness, contrast = item.brightness, item.contrast
        elif item.target_type == "label" and label_key == item.target_value:
            brightness, contrast = item.brightness, item.contrast
        elif item.target_type == "index" and channel_index == item.target_value:
            brightness, contrast = item.brightness, item.contrast
    return brightness, contrast


def sanitize_suffix_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name.strip())
    return cleaned.strip("_") or "label"


def main() -> None:
    args = parse_args()
    normalize_explicit = any(
        token == "--normalize" or token.startswith("--normalize=") for token in sys.argv[1:]
    )
    has_channel_bc = len(args.bc_channel) > 0
    has_global_bc = (float(args.bc_brightness) != 50.0) or (float(args.bc_contrast) != 50.0)
    if (has_channel_bc or has_global_bc) and (not normalize_explicit):
        args.normalize = "minmax"
        print("Info: B/C controls detected without explicit --normalize. Using --normalize minmax (ImageJ-like).")

    input_path: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    label_specs: List[LabelSpec] = []
    for i, spec in enumerate(args.label):
        name, index, color = parse_label_spec(spec)
        if color is None:
            if i >= len(DEFAULT_LABEL_COLORS):
                raise ValueError("Too many labels without explicit colors; please specify red/green/blue")
            color = DEFAULT_LABEL_COLORS[i]
        label_specs.append(LabelSpec(name=name, index=index, color=color))
    label_suffix = f"_{sanitize_suffix_name(label_specs[0].name)}" if label_specs else ""
    channel_bc_overrides = build_channel_bc_overrides(args.bc_channel, label_specs)

    files = iter_tiff_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {input_path}")

    for tif_path in files:
        arr, axes = load_tiff_with_axes(tif_path)
        frames = reorder_to_thwc(arr, axes)
        num_frames, _, _, c = frames.shape

        if args.brightfield < 0 or args.brightfield >= c:
            raise IndexError(
                f"Brightfield channel index {args.brightfield} out of range for {tif_path.name} (C={c})"
            )

        for spec in label_specs:
            if spec.index < 0 or spec.index >= c:
                raise IndexError(
                    f"Label '{spec.name}' channel index {spec.index} out of range for {tif_path.name} (C={c})"
                )
        for item in channel_bc_overrides:
            if item.target_type == "index" and (item.target_value < 0 or item.target_value >= c):
                raise IndexError(
                    f"--bc-channel index {item.target_value} out of range for {tif_path.name} (C={c})"
                )

        bf_bc = resolve_channel_bc(
            channel_index=args.brightfield,
            is_brightfield=True,
            label_name=None,
            default_brightness=args.bc_brightness,
            default_contrast=args.bc_contrast,
            overrides=channel_bc_overrides,
        )
        label_bcs = [
            resolve_channel_bc(
                channel_index=spec.index,
                is_brightfield=False,
                label_name=spec.name,
                default_brightness=args.bc_brightness,
                default_contrast=args.bc_contrast,
                overrides=channel_bc_overrides,
            )
            for spec in label_specs
        ]

        p_low, p_high = resolve_percentiles(
            args.normalize, args.p_low, args.p_high, args.imagej_sat
        )

        if args.normalize != "none" and args.normalize_scope == "global":
            stats = compute_channel_stats(
                frames,
                method=args.normalize,
                p_low=p_low,
                p_high=p_high,
                stat_stride=max(1, args.stat_stride),
                max_samples=max(1, args.max_samples),
            )
            bf_stats = stats[args.brightfield]
            label_stats = [stats[spec.index] for spec in label_specs]
        else:
            bf_stats = None
            label_stats = None

        if args.output_structure == "mirror" and input_path.is_dir():
            rel = tif_path.relative_to(input_path)
            out_path = (output_dir / rel).with_name(f"{rel.stem}{label_suffix}.mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = output_dir / f"{tif_path.stem}{label_suffix}.mp4"

        writer = imageio.get_writer(
            out_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        try:
            for ti in range(num_frames):
                frame = frames[ti]
                brightfield = frame[..., args.brightfield]
                if args.normalize != "none" and args.normalize_scope == "frame":
                    channels = [args.brightfield] + [spec.index for spec in label_specs]
                    frame_stats = compute_frame_stats(
                        frame,
                        channels,
                        method=args.normalize,
                        p_low=p_low,
                        p_high=p_high,
                        max_samples=max(1, args.max_samples),
                    )
                    bf_stats = frame_stats[0]
                    label_stats = frame_stats[1:]
                if label_specs:
                    labels_for_frame: List[LabelSpec] = []
                    for spec in label_specs:
                        labels_for_frame.append(
                            LabelSpec(
                                name=spec.name,
                                index=spec.index,
                                color=spec.color,
                                image=frame[..., spec.index],
                            )
                        )

                    rgb = build_rgb(
                        brightfield,
                        labels_for_frame,
                        args.normalize,
                        bf_stats,
                        label_stats,
                        bf_bc,
                        label_bcs,
                    )
                else:
                    bf_brightness, bf_contrast = bf_bc
                    bf_u8 = scale_to_uint8(
                        brightfield,
                        args.normalize,
                        bf_stats,
                        bf_brightness,
                        bf_contrast,
                    )
                    rgb = np.stack([bf_u8, bf_u8, bf_u8], axis=-1)

                writer.append_data(np.ascontiguousarray(rgb))
        finally:
            writer.close()

    print(f"Processed {len(files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
