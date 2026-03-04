#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/Files/Dendritic_Cells/Migration")
DEFAULT_GLOB = "*_pv.csv"
SOURCE_COL = "FULL_PATH_TO_RAW_DATA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate *_pv.csv files into one CSV and append FULL_PATH_TO_RAW_DATA per segmented-object row."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to search recursively for source CSV files.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_GLOB,
        help="Glob pattern used under root (recursive).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/all_pv_segmented.csv",
    )
    return p.parse_args()


def source_path_for_row(csv_path: Path) -> str:
    # Each source CSV is generated from a same-stem video in the same folder.
    source = csv_path.with_suffix(".mp4").resolve()
    return str(source)


def aggregate_csvs(root: Path, pattern: str, output_path: Path) -> tuple[int, int]:
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found under {root} matching pattern '{pattern}'")

    base_header: list[str] | None = None
    written_rows = 0
    processed_files = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = None

        for csv_file in files:
            with csv_file.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    continue

                if base_header is None:
                    base_header = list(reader.fieldnames)
                    if SOURCE_COL not in base_header:
                        base_header.append(SOURCE_COL)
                    writer = csv.DictWriter(out_f, fieldnames=base_header)
                    writer.writeheader()
                else:
                    incoming = list(reader.fieldnames)
                    base_no_src = [h for h in base_header if h != SOURCE_COL]
                    if incoming != base_no_src and incoming != base_header:
                        raise ValueError(
                            f"Header mismatch in {csv_file}. Expected {base_no_src} but found {incoming}"
                        )

                src_path = source_path_for_row(csv_file)
                for row in reader:
                    # Keep only segmented-object rows (non-empty rows).
                    if not any((v or "").strip() for v in row.values()):
                        continue
                    row[SOURCE_COL] = src_path
                    writer.writerow(row)
                    written_rows += 1

            processed_files += 1

    return processed_files, written_rows


def main() -> int:
    args = parse_args()
    root = args.root
    output = args.output if args.output is not None else root / "all_pv_segmented.csv"

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    files, rows = aggregate_csvs(root=root, pattern=args.pattern, output_path=output)
    print(f"Aggregated {files} file(s) into: {output}")
    print(f"Total segmented-object rows: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
