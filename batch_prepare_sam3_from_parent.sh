#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <parent_dir> <output_root> <brightfield_channel> [extra args...]" >&2
  echo "Example:" >&2
  echo "  $0 /mnt/Files/Dendritic_Cells /mnt/Files/Dendritic_Cells/Migration 0 --label gfp:3:green" >&2
  exit 1
fi

PARENT_DIR="$1"
OUTPUT_ROOT="$2"
BRIGHTFIELD="$3"
shift 3

if [ ! -d "$PARENT_DIR" ]; then
  echo "Error: parent_dir does not exist: $PARENT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

count=0
while IFS= read -r -d '' tif; do
  src_dir="$(dirname "$tif")"
  subfolder="$(basename "$src_dir")"
  out_dir="$OUTPUT_ROOT/$subfolder"
  mkdir -p "$out_dir"

  python /home/jianfei/SegmentAnything/DendriticCellTest/prepare_sam3_data.py \
    --input "$tif" \
    --output "$out_dir" \
    --brightfield "$BRIGHTFIELD" \
    "$@"

  count=$((count + 1))
done < <(
  find "$PARENT_DIR" -type f \( -name 'WT*.tif' -o -name 'MyoKO*.tif' \) ! -path "$OUTPUT_ROOT/*" -print0
)

echo "Processed $count file(s) into $OUTPUT_ROOT"
