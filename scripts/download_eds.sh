#!/bin/bash

# Script to download and extract EDS dataset sequences
# Usage: ./download_eds_data.sh [output_directory]

set -e  # Exit on error

OUTPUT_DIR="$(pwd)/data/eds"
if [ "$1" != "" ]; then
    OUTPUT_DIR="$1"
fi

mkdir -p "$OUTPUT_DIR"
echo "Downloading datasets to: $OUTPUT_DIR"

BASE_URL="https://download.ifi.uzh.ch/rpg/eds/dataset"

SEQUENCES=(
    "01_peanuts_light"
    "02_rocket_earth_light"
    "08_peanuts_running"
    "14_ziggy_in_the_arena"
)

for seq in "${SEQUENCES[@]}"; do
    echo "======================================================="
    echo "Processing sequence: $seq"
    
    mkdir -p "$OUTPUT_DIR/$seq"
    
    URL="$BASE_URL/$seq/$seq.tgz"
    TGZ_FILE="$OUTPUT_DIR/$seq/$seq.tgz"
    
    echo "Downloading from: $URL"
    wget -c "$URL" -O "$TGZ_FILE"
    
    echo "Extracting..."
    tar -xzf "$TGZ_FILE" -C "$OUTPUT_DIR/$seq"
    
    echo "Removing archive..."
    rm "$TGZ_FILE"
    
    echo "Done with $seq"
    echo ""
done

echo "All sequences have been downloaded and extracted."
echo "Data is available in: $OUTPUT_DIR"

echo "======================================================="
echo "Summary of downloaded data:"
for seq in "${SEQUENCES[@]}"; do
    echo "$seq:"
    ls -la "$OUTPUT_DIR/$seq" | grep -v "^total"
    echo ""
done