import os
from collections import OrderedDict

import cv2

RADIUS = 255

STANDARD_LINES = OrderedDict({
    # "sideLine1": [
    #     (0.0035, 0.01),
    #     (0.0035, 0.90), 
    # ],
    "endLine2": [
        (0.065, 0.01), 
        (0.935, 0.01)
    ],
    # "sideLine3": [
    #     (0.9914, 0.0087), (0.9914, 0.2029), (0.9948, 0.3942),
    #     (0.9948, 0.5942), (0.9914, 0.7681), (0.9914, 0.9304),
    # ],
    "threePointLine4": [
            (0.065, 0.34),
            (0.8692, 0.6029),
            (0.50, 0.93),
            (0.935, 0.34),

    ],
    "laneLine5": [
        (0.34, 0.01), (0.34, 0.650),
    ],
    "laneLine6": [
        (0.661, 0.01),(0.661, 0.650), 
    ],
    "center_point":[
        (0.50, 0.195),
    ]
    # "freeThrowLine7": [
    #     (0.3408, 0.6522), (0.3838, 0.6522), (0.4527, 0.6551),
    #     (0.5387, 0.6522), (0.6127, 0.6551), (0.6627, 0.6551),
    # ],
    # "freeThrowCircle8": [
    #     (0.3838, 0.6522), (0.4010, 0.7565), (0.4355, 0.8203),
    #     (0.5009, 0.8551), (0.5645, 0.8261), (0.6041, 0.7478),
    #     (0.6162, 0.6522),
    # ],
    # "fieldFilterLines": [
    #     (0.0017, 0.9014), (0.0017, 0.7188), (0.0017, 0.4638), (0.0017, 0.2058),
    #     (0.0000, 0.0029), (0.1480, 0.0029), (0.3029, 0.0087), (0.4664, 0.0087),
    #     (0.6213, 0.0029), (0.7625, 0.0087), (0.8795, 0.0087), (0.9914, 0.0087),
    #     (0.9914, 0.2174), (0.9914, 0.4493), (0.9914, 0.6667), (0.9914, 0.8986),
    # ],
})


def norm_to_pixel(pt, width, height):
    return int(round(pt[0] * width)), int(round(pt[1] * height))


def clamp_text_origin(text, origin, font_face, font_scale, thickness, image_width, image_height):
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width, text_height = text_size

    x, y = origin
    x = max(0, min(x, max(0, image_width - text_width - 1)))
    y = max(text_height + baseline, min(y, max(text_height + baseline, image_height - baseline - 1)))
    return x, y


def draw_points(image, lines):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    palette = [
        (0, 0, 255),
        (0, 165, 255),
        (128, 0, 255),
        (255, 0, 0),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ]

    for line_index, (line_name, points) in enumerate(lines.items(), start=1):
        color = palette[(line_index - 1) % len(palette)]
        print(f"{line_name}:")
        for point_index, pt in enumerate(points, start=1):
            x, y = norm_to_pixel(pt, width, height)
            print(f"  {point_index:02d}: ({x}, {y})  ({pt[0]*width:.3f}, {pt[1]*height:.3f})")
            cv2.circle(canvas, (x, y), 3, color, -1)
            label_text = f"{line_index}-{point_index}"
            norm_text = f"({pt[0]:.3f}, {pt[1]:.3f})"

            label_origin = clamp_text_origin(
                label_text,
                (x + 8, y + 3),
                font_face,
                0.45,
                2,
                width,
                height,
            )
            norm_origin = clamp_text_origin(
                norm_text,
                (x - 40, y + 30),
                font_face,
                0.35,
                1,
                width,
                height,
            )

            cv2.putText(
                canvas,
                label_text,
                label_origin,
                font_face,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                norm_text,
                norm_origin,
                font_face,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "python", "base", "base_standard.png")
    output_path = os.path.join(base_dir, "python", "base", "base_standard_annotated.png")

    image = cv2.imread(image_path)
    # 绘制圆弧
    int_centerx = int(STANDARD_LINES['center_point'][0][0] * image.shape[1])
    int_centery = int(STANDARD_LINES['center_point'][0][1] * image.shape[0])
    cv2.circle(image, (int_centerx, int_centery), RADIUS, (255, 0, 255), 2, cv2.LINE_AA)
    # 将半径给打印出来
    cv2.putText(
        image,
        f"R={RADIUS}",
        (int_centerx, int_centery + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    print(f"image_path: {image_path}")
    print(f"image_shape: {image.shape}")

    annotated = draw_points(image, STANDARD_LINES)
    cv2.imwrite(output_path, annotated)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()