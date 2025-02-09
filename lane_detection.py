import os
import sys
import numpy as np
from simple_image_lib import SimpleImageLib
from collections import Counter

def compute_histogram(image):
    flat_pixels = [pixel for row in image for pixel in row]
    return Counter(flat_pixels)

def dynamic_threshold(image):
    hist = compute_histogram(image)
    min_val = min(hist)
    max_val = max(hist)
    return (min_val + max_val) // 2

def detect_lanes(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' does not exist.")
        sys.exit(1)

    print("Reading the image...")
    image = SimpleImageLib.read_image(filepath)
    print(f"Image size: {len(image)}x{len(image[0])}")

    resized_image = SimpleImageLib.resize(image, 100, 50)
    print(f"Resized image size: {len(resized_image)}x{len(resized_image[0])}")

    threshold = dynamic_threshold(resized_image)
    print(f"Using dynamic threshold: {threshold}")

    binary_image = SimpleImageLib.grayscale_threshold(resized_image, threshold)
    print("Binary image (thresholded):")
    print_image(binary_image)

    edges = SimpleImageLib.detect_edges(binary_image)
    print("Edges detected:")
    print_image(edges)

    return edges

def print_image(image):
    for row in image:
        print("".join("█" if pixel > 128 else " " for pixel in row))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lane_detection.py <image_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    detected_lanes = detect_lanes(filepath)
    print("Detected Lanes:")
    print_image(detected_lanes)
