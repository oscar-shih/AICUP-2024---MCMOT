#!/bin/bash

DEFAULT_WEIGHTS="models/yolov8.pt"
DEFAULT_SOURCE_DIR="../datasets/32_33_AI_CUP_testdataset/AI_CUP_testdata/images"
DEFAULT_DEVICE="0"
DEFAULT_FAST_REID_CONFIG="models/bagtricks_R50-ibn.yml"
DEFAULT_FAST_REID_WEIGHTS="models/fast_reid.pth"
DEFAULT_SAVE_DIR="submission"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --weights)
            WEIGHTS="$2"
            shift
            shift
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift
            shift
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --fast-reid-config)
            FAST_REID_CONFIG="$2"
            shift
            shift
            ;;
        --fast-reid-weights)
            FAST_REID_WEIGHTS="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Using default values if not provided
WEIGHTS="${WEIGHTS:-$DEFAULT_WEIGHTS}"
SOURCE_DIR="${SOURCE_DIR:-$DEFAULT_SOURCE_DIR}"
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
FAST_REID_CONFIG="${FAST_REID_CONFIG:-$DEFAULT_FAST_REID_CONFIG}"
FAST_REID_WEIGHTS="${FAST_REID_WEIGHTS:-$DEFAULT_FAST_REID_WEIGHTS}"
SAVE_DIR="${SAVE_DIR:-$DEFAULT_SAVE_DIR}"

for folder in "$SOURCE_DIR"/*; do
    timestamp=${folder##*/}
    python3 demo.py --weights "$WEIGHTS" --source "$folder" --device "$DEVICE" --name "$timestamp" --fuse-score --agnostic-nms --with-reid --fast-reid-config "$FAST_REID_CONFIG" --fast-reid-weights "$FAST_REID_WEIGHTS" --project "$SAVE_DIR"
    # break
done

