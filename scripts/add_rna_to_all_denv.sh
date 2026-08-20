#!/bin/bash
# Add RNA structure to all dengue mutation count files

set -e

DENGUE_DIR="/home/sasha/git_repos/denguedata"
RNA_DIR="/home/sasha/Downloads/tmp"
SCRIPT="scripts/add_rna_structure.py"

for SEROTYPE in 1 2 3 4; do
    for REGION in genome E; do
        echo "Processing DENV${SEROTYPE} ${REGION}..."
        
        INPUT_DIR="${DENGUE_DIR}/denv${SEROTYPE}/${REGION}"
        MUT_COUNTS="${INPUT_DIR}/mut_counts_by_clade.csv"
        RNA_FILE="${RNA_DIR}/DENV${SEROTYPE}_rna_structure.txt"
        OUTPUT="${INPUT_DIR}/mut_counts_by_clade_with_rna.csv"
        
        if [ -f "$MUT_COUNTS" ] && [ -f "$RNA_FILE" ]; then
            python "$SCRIPT" \
                --counts "$MUT_COUNTS" \
                --structure "$RNA_FILE" \
                --output "$OUTPUT"
            
            # Replace original file with RNA-annotated version
            mv "$OUTPUT" "$MUT_COUNTS"
            echo "  ✓ Added unpaired column to ${MUT_COUNTS}"
        else
            echo "  ✗ Missing files for DENV${SEROTYPE} ${REGION}"
            [ ! -f "$MUT_COUNTS" ] && echo "    Missing: $MUT_COUNTS"
            [ ! -f "$RNA_FILE" ] && echo "    Missing: $RNA_FILE"
        fi
    done
done

echo "Done!"
