from PIL import Image
import os
import sys

class SimpleImageLib:
    @staticmethod
    def read_image(filepath):
        """Read an image file and convert it into a 2D grayscale array."""
        with Image.open(filepath) as img:
            img = img.convert("L")  # Convert to grayscale (L mode)
            return [[img.getpixel((x, y)) for x in range(img.width)] for y in range(img.height)]

    @staticmethod
    def resize(image, new_width, new_height):
        """Resize the image using nearest-neighbor interpolation."""
        old_height = len(image)
        old_width = len(image[0])
        scale_x = old_width / new_width
        scale_y = old_height / new_height

        resized = [
            [
                image[int(y * scale_y)][int(x * scale_x)]
                for x in range(new_width)
            ]
            for y in range(new_height)
        ]
        return resized

    @staticmethod
    def grayscale_threshold(image, threshold):
        """Convert an image to binary based on a grayscale threshold."""
        return [
            [255 if pixel > threshold else 0 for pixel in row]  # 255 for white, 0 for black
            for row in image
        ]

    @staticmethod
    def detect_edges(image):
        """Simple edge detection using a Sobel-like kernel."""
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        height, width = len(image), len(image[0])
        edges = [[0 for _ in range(width)] for _ in range(height)]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gx = sum(
                    kernel_x[ky][kx] * image[y + ky - 1][x + kx - 1]
                    for ky in range(3) for kx in range(3)
                )
                gy = sum(
                    kernel_y[ky][kx] * image[y + ky - 1][x + kx - 1]
                    for ky in range(3) for kx in range(3)
                )
                edges[y][x] = min(255, int((gx ** 2 + gy ** 2) ** 0.5))  # Compute the magnitude of the gradient

        return edges

    @staticmethod
    def print_image(image):
        """Utility to display the image matrix in the console."""
        for row in image:
            print(" ".join(str(pixel) for pixel in row))

def compute_histogram(image):
    """Compute the histogram of pixel intensities in the image."""
    flat_pixels = [pixel for row in image for pixel in row]
    return flat_pixels

def dynamic_threshold(image):
    """Automatically determine a threshold based on image histogram."""
    hist = compute_histogram(image)
    min_val = min(hist)
    max_val = max(hist)
    return (min_val + max_val) // 2  # Basic midpoint for thresholding

def detect_lanes(filepath):
    if not os.path.exists(filepath):  # Check if the file exists
        print(f"Error: The file '{filepath}' does not exist.")
        sys.exit(1)

    # 1. Read the image (already grayscale)
    print("Reading the image...")
    image = SimpleImageLib.read_image(filepath)  # Image is automatically converted to grayscale
    print(f"Image size: {len(image)}x{len(image[0])} (height x width)")

    # 2. Resize the image to make processing faster
    resized_image = SimpleImageLib.resize(image, 100, 50)
    print(f"Resized image size: {len(resized_image)}x{len(resized_image[0])}")

    # 3. Compute dynamic threshold based on the histogram of the image
    threshold = dynamic_threshold(resized_image)
    print(f"Using dynamic threshold: {threshold}")

    # Convert to binary (lane detection often works on high contrast)
    binary_image = SimpleImageLib.grayscale_threshold(resized_image, threshold)
    print("Binary image (thresholded):")
    SimpleImageLib.print_image(binary_image)

    # 4. Detect edges in the binary image
    edges = SimpleImageLib.detect_edges(binary_image)
    print("Edges detected:")
    SimpleImageLib.print_image(edges)

    # 5. Return the detected edges (lanes)
    return edges

if __name__ == "__main__":
    # Use the first argument as the image path
    if len(sys.argv) < 2:
        print("Usage: python lane_detection.py <image_path>")
        sys.exit(1)

    filepath = sys.argv[1]
    detected_lanes = detect_lanes(filepath)
    print("Detected Lanes:")
    SimpleImageLib.print_image(detected_lanes)
