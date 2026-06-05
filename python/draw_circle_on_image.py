import argparse
import os

import cv2


def draw_circle(image, center_norm, radius):
    height, width = image.shape[:2]
    center_x = int(round(center_norm[0] * width))
    center_y = int(round(center_norm[1] * height))
    radius_px = radius

    canvas = image.copy()
    cv2.circle(canvas, (center_x, center_y), radius_px, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (center_x, center_y), 3, (0, 255, 0), -1, cv2.LINE_AA)
    return canvas, (center_x, center_y), radius_px


def main():
    parser = argparse.ArgumentParser(description="Draw a circle on an image using normalized coordinates.")
    parser.add_argument("--image", default=os.path.join("python", "base", "base_standard.png"), help="Input image path")
    parser.add_argument("--output", default=os.path.join("python", "base", "base_standard_circle.png"), help="Output image path")
    parser.add_argument("--cx", type=float, default=0.50, help="Normalized circle center x")
    parser.add_argument("--cy", type=float, default=0.195, help="Normalized circle center y")
    parser.add_argument("--radius", type=float, default=255, help="Normalized circle radius")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    drawn, center_px, radius_px = draw_circle(image, (args.cx, args.cy), args.radius)
    ok = cv2.imwrite(args.output, drawn)
    if not ok:
        raise RuntimeError(f"Unable to write image: {args.output}")

    print(f"image: {args.image}")
    print(f"output: {args.output}")
    print(f"center_norm: ({args.cx:.3f}, {args.cy:.3f})")
    print(f"radius_norm: {args.radius:.3f}")
    print(f"center_px: {center_px}")
    print(f"radius_px: {radius_px}")


if __name__ == "__main__":
    main()
