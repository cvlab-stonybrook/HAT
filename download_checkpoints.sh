#!/bin/bash

# List of TAP variants
TAP_LIST=("FV" "TP" "TA")

# Create checkpoints directory if it doesn't exist
mkdir -p ./checkpoints

# Download HAT checkpoints
for TAP in "${TAP_LIST[@]}"; do
    FILE="./checkpoints/HAT_${TAP}.pt"
    if [ ! -f "$FILE" ]; then
        echo "Downloading model checkpoint for TAP=${TAP}..."
        wget "http://vision.cs.stonybrook.edu/~cvlab_download/HAT/HAT_${TAP}.pt" -P ./checkpoints/
    else
        echo "Checkpoint HAT_${TAP}.pt already exists. Skipping..."
    fi
done

# Create pretrained_models directory if it doesn't exist
mkdir -p ./pretrained_models

# Download pretrained model weights if not already present
if [ ! -f "./pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl" ]; then
    echo "Downloading pretrained model weights..."
    wget "http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl" -P ./pretrained_models/
    wget "http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl" -P ./pretrained_models/
else
    echo "Pretrained models already exist. Skipping..."
fi
