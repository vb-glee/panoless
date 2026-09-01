import cv2
import numpy as np
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(_root, "envs")
OUTPUT_DIR = os.path.join(_root, "envs")
TARGET_WIDTH = 1025
TARGET_HEIGHT = 512


def convert_hdr_to_png(hdr_path, output_path, width=TARGET_WIDTH, height=TARGET_HEIGHT):
    # Read HDR image (loaded as float32 BGR by OpenCV)
    hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if hdr is None:
        print(f"  ERROR: Could not read {hdr_path}")
        return False

    print(f"  Original shape: {hdr.shape}, dtype: {hdr.dtype}, range: [{hdr.min():.4f}, {hdr.max():.4f}]")

    # Tone mapping using Reinhard (preserves colors well)
    tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0, light_adapt=0.8, color_adapt=0.0)
    ldr = tonemap.process(hdr)

    # Clamp to [0, 1] and convert to 8-bit
    ldr = np.clip(ldr, 0.0, 1.0)
    ldr_8bit = (ldr * 255).astype(np.uint8)

    # Resize to target resolution (width x height)
    resized = cv2.resize(ldr_8bit, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # Flip on vertical axis (mirror left-right)
    resized = cv2.flip(resized, 1)

    # Save as PNG
    cv2.imwrite(output_path, resized)
    print(f"  Saved: {output_path} ({width}x{height})")
    return True


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    hdr_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".hdr")]
    if not hdr_files:
        print("No .hdr files found.")
        return

    print(f"Found {len(hdr_files)} HDR file(s)\n")
    for fname in sorted(hdr_files):
        hdr_path = os.path.join(input_dir, fname)
        png_name = os.path.splitext(fname)[0] + ".png"
        output_path = os.path.join(output_dir, png_name)
        print(f"Converting: {fname}")
        convert_hdr_to_png(hdr_path, output_path)
        print()

    print("Done!")


if __name__ == "__main__":
    main()