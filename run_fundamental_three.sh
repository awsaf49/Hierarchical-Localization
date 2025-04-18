#!/bin/bash

# If no arguments provided, use default images from dataset/front folder
if [ "$#" -eq 0 ]; then
    FRONT_DIR="datasets/front-door"
    # Get all JPG files from the directory (case insensitive)
    IMAGES=($(find $FRONT_DIR -type f -iname "*.jpg"))
    
    if [ ${#IMAGES[@]} -lt 3 ]; then
        echo "Error: Less than 3 images found in $FRONT_DIR"
        exit 1
    fi
    
    set -- "${IMAGES[0]}" "${IMAGES[1]}" "${IMAGES[2]}"
    echo "Using default images: $1 $2 $3"
elif [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_A> <image_B> <image_C>"
    exit 1
fi

for img in "$@"; do
    [ ! -f "$img" ] && echo "Error: $img not found" && exit 1
done

python3 FundamentalMatrix3images.py "$1" "$2" "$3"
