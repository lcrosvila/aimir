#!/bin/bash

# Define the three folders to search
folder1="/home/laura/aimir/suno/audio/embeddings/clap-laion-music"
folder2="/home/laura/aimir/suno/audio/embeddings/clap-laion-music"
# folder3="/home/laura/aimir/suno/audio"

# Find .mp3 and .json files in the specified folders and calculate total size
total_size=$(find "$folder1" "$folder2" -type f \( -name "*.npy" -o -name "*.json" \) -print0 | du -ch --files0-from=- | grep total$ | cut -f1)

echo "Total size of .mp3 and .json files in the specified folders: $total_size"