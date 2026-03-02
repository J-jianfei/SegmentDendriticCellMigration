#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <input_dir> <output_dir> <brightfield_channel> [extra args...]" >&2
  echo "Example:" >&2
  echo "  $0 ./tifs ./mp4 0 --label gfp:3:green --label parasite:2:red --normalize-scope frame" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
BRIGHTFIELD="$3"
shift 3

python /home/jianfei/SegmentAnything/DendriticCellTest/prepare_sam3_data.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --brightfield "$BRIGHTFIELD" \
  "$@"
