import math
from pathlib import Path
import cv2
import numpy as np


def compute_three_point_status(abs_point, include_line=True, eps=10):
    """仅计算点相对于三分线的位置，

    返回字典：{"outside", "on_line", "region", "distance_to_center"}
    """

    # 三分线参数
    cx, cy = 290.500, 67.275       # 三分线圆心
    left_x = 37.765               # 左侧底角三分线
    right_x = 543.235             # 右侧底角三分线
    corner_y = 117.300            # 直线段与圆弧连接处 y
    radius = 255                  # 圆弧半径（近似）

    on_line = False
    distance_to_center = None

    x, y = abs_point
    if y <= corner_y:
        region = "corner_line_area"
        on_left_line = abs(x - left_x) <= eps
        on_right_line = abs(x - right_x) <= eps
        on_line = on_left_line or on_right_line

        if include_line:
            outside = x <= left_x + eps or x >= right_x - eps
        else:
            outside = x < left_x - eps or x > right_x + eps
    else:
        region = "arc_area"
        distance_to_center = math.hypot(x - cx, y - cy)
        print(f"distance_to_center: {distance_to_center:.3f}, radius: {radius:.3f}")
        on_line = abs(distance_to_center - radius) <= eps

        if include_line:
            outside = distance_to_center >= radius - eps
        else:
            outside = distance_to_center > radius + eps

    return {
        "outside": outside,
        "on_line": on_line,
        "region": region,
        "distance_to_center": distance_to_center,
    }


def draw_point_on_image_cv(
    image_path,
    x,
    y,
    status: dict,
    output_path=None,
    show=True,
):
    """使用 OpenCV 在图片上绘制点和文字说明。返回保存路径或 None。"""

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"image not found: {image_path}")

    # BGR colors
    if status.get("on_line"):
        color = (0, 165, 255)  # 橙色 (B,G,R)
        label = "in line"
    elif status.get("outside"):
        color = (0, 0, 255)    # 红色
        label = "outside"
    else:
        color = (0, 180, 0)    # 绿色
        label = "inside"

    px = int(round(x))
    py = int(round(y))

    # 画点（实心圆）和白色边框
    cv2.circle(img, (px, py), 8, color, -1)
    cv2.circle(img, (px, py), 8, (255, 255, 255), 2)

    text = f"({x:.1f}, {y:.1f}) {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 默认文字在点右上方
    text_x = px + 12
    text_y = py - 12

    h, w = img.shape[:2]
    if text_x + text_w + 8 > w:
        text_x = px - (12 + text_w)
    if text_y - text_h - 8 < 0:
        text_y = py + 12 + text_h

    # 背景框
    rect_tl = (text_x - 4, text_y - text_h - 4)
    rect_br = (text_x + text_w + 4, text_y + baseline + 4)
    cv2.rectangle(img, rect_tl, rect_br, (255, 255, 255), -1)
    cv2.rectangle(img, rect_tl, rect_br, color, 1)

    # 写文字（BGR）
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # 画一条从文字到点的线
    line_end_y = text_y - int(text_h / 2)
    cv2.line(img, (px, py), (text_x, line_end_y), color, 1, cv2.LINE_AA)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_name(p.stem + "_with_point" + p.suffix))

    cv2.imwrite(str(output_path), img)

    if show:
        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path


def judge_three_point_area(
    abs_point,
    image_path=None,
    output_path=None,
    include_line=True,
    eps=1e-6,
    show=True,
):
    """兼容旧接口：先计算再（可选）使用 OpenCV 绘图。

    主要逻辑已分离到 compute_three_point_status 和 draw_point_on_image_cv。
    """

    status = compute_three_point_status(abs_point, include_line=include_line, eps=eps)
    saved_path = None
    if image_path is not None:
        saved_path = draw_point_on_image_cv(image_path, abs_point[0], abs_point[1], status, output_path=output_path, show=show)

    status["output_path"] = saved_path
    return status


if __name__ == "__main__":
    # 示例 1：只判断，不画图
    # result = judge_three_point_area(290.5, 300)
    # print(result)
    # print("是否在三分线外：", result["outside"])
    test_point = (290.5, 300)
    # 示例 2：判断并绘制到图片上
    result = judge_three_point_area(
        abs_point=test_point,
        image_path="python/base/base_standard_annotated.png",
        output_path="tmp.png",
        include_line=True,
        show=True,
    )

    print(result)
    print("是否在三分线外：", result["outside"])
    print("结果图路径：", result["output_path"])