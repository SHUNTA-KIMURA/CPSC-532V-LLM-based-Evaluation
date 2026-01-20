#!/bin/bash

set -e

INPUT_DIR="wiki"
OUTPUT_DIR="summary"
SCRIPT="src/summarize.py"

mkdir -p "$OUTPUT_DIR"

for infile in "$INPUT_DIR"/*.txt; do
  base=$(basename "$infile" .txt)
  outfile="$OUTPUT_DIR/${base}_summary.txt"

  echo "Processing: $infile -> $outfile"

  python "$SCRIPT" \
    --in_txt "$infile" \
    --out_txt "$outfile"
done

echo "All files processed."