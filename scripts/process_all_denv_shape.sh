#!/bin/bash
# Process all DENV SHAPE files to unpaired format

set -e

SHAPE_DIR="/home/sasha/Downloads/tmp"
OUTPUT_DIR="/home/sasha/Downloads/tmp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Processing DENV SHAPE files..."
echo "Input directory: $SHAPE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# DENV1
python3 "$SCRIPT_DIR/process_shape_to_unpaired.py" \
    --input "$SHAPE_DIR/GSM7086062_DENV1_exvirion_shapemapper-output.txt" \
    --output "$OUTPUT_DIR/DENV1_rna_structure.txt" \
    --threshold 0.4

# DENV2
python3 "$SCRIPT_DIR/process_shape_to_unpaired.py" \
    --input "$SHAPE_DIR/GSM7086064_DENV2_exvirion_shapemapper-output.txt" \
    --output "$OUTPUT_DIR/DENV2_rna_structure.txt" \
    --threshold 0.4

# DENV3
python3 "$SCRIPT_DIR/process_shape_to_unpaired.py" \
    --input "$SHAPE_DIR/GSM7086066_DENV3_exvirion_shapemapper-output.txt" \
    --output "$OUTPUT_DIR/DENV3_rna_structure.txt" \
    --threshold 0.4

# DENV4
python3 "$SCRIPT_DIR/process_shape_to_unpaired.py" \
    --input "$SHAPE_DIR/GSM7086068_DENV4_exvirion_shapemapper-output.txt" \
    --output "$OUTPUT_DIR/DENV4_rna_structure.txt" \
    --threshold 0.4

echo ""
echo "All DENV SHAPE files processed!"
