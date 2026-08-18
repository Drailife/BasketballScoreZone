# 使用 (250, 250, 250) 填充扩充区域。
# 分别支持上下左右不同的扩充尺寸。
# 保存扩充后的图片。
# 输出并保存重新计算的 A–F 六个归一化点位 JSON。
import argparse
from pathlib import Path

import cv2


STANDARD_LINES = {
    "A": (0.065, 0.010),
    "B": (0.340, 0.010),
    "C": (0.661, 0.010),
    "D": (0.935, 0.010),
    "E": (0.340, 0.650),
    "F": (0.661, 0.650),
}


def expand_court_image(input_path, output_path, left=0, right=0, top=0, bottom=0):
    padding = (left, right, top, bottom)
    if any(value < 0 for value in padding):
        raise ValueError("扩充像素数不能为负数")

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    old_height, old_width = image.shape[:2]
    new_width = old_width + left + right
    new_height = old_height + top + bottom
    expanded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(250, 250, 250),
    )

    new_points = {
        label: (
            round((x * old_width + left) / new_width, 6),
            round((y * old_height + top) / new_height, 6),
        )
        for label, (x, y) in STANDARD_LINES.items()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), expanded):
        raise OSError(f"无法保存图片: {output_path}")

    return new_points, (new_width, new_height)


def parse_args():
    default_input = Path(__file__).resolve().parent / "data" / "base_standard.png"
    parser = argparse.ArgumentParser(
        description="扩充篮球场图片并重新计算 A-F 六个归一化点位"
    )
    parser.add_argument("--input", type=Path, default=default_input, help="输入图片")
    parser.add_argument("--output", type=Path, help="输出图片")
    parser.add_argument("--left", type=int, default=0, help="左侧扩充像素数")
    parser.add_argument("--right", type=int, default=0, help="右侧扩充像素数")
    parser.add_argument("--top", type=int, default=0, help="上方扩充像素数")
    parser.add_argument("--bottom", type=int, default=0, help="下方扩充像素数")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_expanded.png")
    points, image_size = expand_court_image(
        args.input,
        output_path,
        left=args.left,
        right=args.right,
        top=args.top,
        bottom=args.bottom,
    )

    formatted_points = ", ".join(
        f'"{label}":({x:.6f}, {y:.6f})' for label, (x, y) in points.items()
    )
    print(f"新图尺寸: {image_size[0]} x {image_size[1]}")
    print(f"STANDARD_LINES={{{formatted_points}}}")
    print(f"图片已保存: {output_path}")


if __name__ == "__main__":
    main()
