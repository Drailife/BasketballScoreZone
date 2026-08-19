import math
import os

import cv2
import numpy as np

STANDARD_LINES = {
    "A": (0.128877, 0.043028),
    "B": (0.363495, 0.043028),
    "C": (0.637358, 0.043028),
    "D": (0.871123, 0.043028),
    "E": (0.363495, 0.448165),
    "F": (0.637358, 0.448165),
    "G": (0.128877, 0.251927),
    "H": (0.871123, 0.251927),
    "Circle Center": (0.500000, 0.160138),
}
RADIUS = 255
image = "python/base/base_standard_exp_681x545.png"


def norm_to_pixel(pt, width, height):
    return int(round(pt[0] * width)), int(round(pt[1] * height))


def put_text(canvas, text, origin, font, scale, color, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = canvas.shape[:2]
    x, y = origin
    x = max(0, min(x, width - text_w - 1))
    y = max(text_h + baseline, min(y, height - baseline - 1))
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, image)
    output_path = os.path.join(
        base_dir, "python", "base", "base_standard_exp_681x545_annotated.png"
    )

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    height, width = img.shape[:2]
    print(f"image_path: {image_path}")
    print(f"image_shape: {img.shape}")

    canvas = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 以 Circle Center 为圆心绘制圆弧：起点为 G，终点为 H（向下凸出）
    center = STANDARD_LINES["Circle Center"]
    g = STANDARD_LINES["G"]
    h = STANDARD_LINES["H"]

    cx = center[0] * width
    cy = center[1] * height
    gx, gy = g[0] * width, g[1] * height
    hx, hy = h[0] * width, h[1] * height

    angle_g = math.atan2(gy - cy, gx - cx)
    angle_h = math.atan2(hy - cy, hx - cx)

    # 半径取圆心到 G/H 的实际距离，保证弧线两端正好落在 G、H 上
    radius = math.hypot(gx - cx, gy - cy)

    num_points = 180
    arc_points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = angle_g + (angle_h - angle_g) * t
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        arc_points.append((x, y))

    cv2.polylines(
        canvas,
        [np.array(arc_points, dtype=np.int32).reshape(-1, 1, 2)],
        False,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    print(
        f"arc: G({gx:.1f}, {gy:.1f}) -> H({hx:.1f}, {hy:.1f}), "
        f"center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
    )

    for name, pt in STANDARD_LINES.items():
        x, y = norm_to_pixel(pt, width, height)
        print(f"{name}: ({x}, {y})")

        coord_text = f"({pt[0]:.3f}, {pt[1]:.3f})"

        if name == "Circle Center":
            color = (0, 255, 0)  # 绿色
            cv2.circle(canvas, (x, y), 6, color, -1, cv2.LINE_AA)
            put_text(canvas, "CC", (x + 10, y - 10), font, 0.5, color, 2)
            put_text(canvas, coord_text, (x + 10, y + 16), font, 0.4, color, 1)
            put_text(canvas, f"R={RADIUS}px", (x + 10, y + 34), font, 0.5, (255, 0, 255), 2)
        else:
            color = (0, 0, 255)  # 红色
            cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
            put_text(canvas, name, (x + 10, y - 10), font, 0.7, color, 2)
            put_text(canvas, coord_text, (x + 10, y + 16), font, 0.4, color, 1)

    cv2.imwrite(output_path, canvas)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
