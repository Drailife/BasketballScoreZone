import argparse
import json
import math
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


KEYPOINT_LABELS = ("A", "B", "C", "D", "E", "F")
TEMPLATE_KEYPOINTS = (
    (40.0, 4.0),
    (199.0, 5.0),
    (386.0, 4.0),
    (543.0, 2.0),
    (199.0, 223.0),
    (384.0, 224.0),
)

MIN_HOMOGRAPHY_POINTS = 4
DEFAULT_KEYPOINT_CONF_THRESHOLD = 0.5
DEFAULT_SCORE_WITHOUT_HOMOGRAPHY = 2

# These constants match python/judge_in_three_line.py and template coordinates.
THREE_LINE_CENTER = (290.5, 67.275)
THREE_LINE_LEFT_X = 37.765
THREE_LINE_RIGHT_X = 543.235
THREE_LINE_CORNER_Y = 117.3
THREE_LINE_RADIUS = 255.0
THREE_LINE_EPS = 10.0

COLOR_TWO_POINT = (0, 180, 0)
COLOR_THREE_POINT = (0, 0, 255)
COLOR_DEFAULT_SCORE = (0, 165, 255)
COLOR_TEXT_BG = (255, 255, 255)

FONT_SCALE = 0.65
FONT_THICKNESS = 2
LABEL_PADDING = 5

Point = Tuple[float, float]
PersonDetection = Dict[str, Any]


def find_homography(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    """
    使用最小二乘法（SVD）计算单应性矩阵。

    src_pts: 原图中的点位，至少 4 个。
    dst_pts: 模板图中的对应点位，数量必须与 src_pts 相同。
    """
    if len(src_pts) != len(dst_pts) or len(src_pts) < MIN_HOMOGRAPHY_POINTS:
        raise ValueError("源点和目标点数量必须相等且至少4个")

    matrix_rows = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        matrix_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])

    _, _, vt = np.linalg.svd(np.asarray(matrix_rows, dtype=np.float64))
    homography = vt[-1].reshape(3, 3)

    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("单应性矩阵归一化失败")

    homography = homography / homography[2, 2]
    if not np.all(np.isfinite(homography)):
        raise ValueError("单应性矩阵包含无效值")

    return homography


def map_point_A_to_warpedA(pt: Point, matrix: np.ndarray) -> Point:
    """
    将原图点位通过单应性矩阵映射到模板图坐标系。
    """
    point = np.asarray([[pt[0]], [pt[1]], [1.0]], dtype=np.float64)
    mapped = matrix @ point

    if abs(mapped[2, 0]) < 1e-12:
        raise ValueError("点位映射归一化失败")

    mapped /= mapped[2, 0]
    return float(mapped[0, 0]), float(mapped[1, 0])


def compute_three_point_status(
    template_point: Point,
    include_line: bool = True,
    eps: float = THREE_LINE_EPS,
) -> Dict[str, Any]:
    """
    判断模板图点位是否位于三分线外。

    outside=True 表示三分线外，按 3 分处理；否则按 2 分处理。
    """
    x, y = template_point
    cx, cy = THREE_LINE_CENTER
    on_line = False
    distance_to_center = None

    if y <= THREE_LINE_CORNER_Y:
        region = "corner_line_area"
        on_line = (
            abs(x - THREE_LINE_LEFT_X) <= eps
            or abs(x - THREE_LINE_RIGHT_X) <= eps
        )

        if include_line:
            outside = x <= THREE_LINE_LEFT_X + eps or x >= THREE_LINE_RIGHT_X - eps
        else:
            outside = x < THREE_LINE_LEFT_X - eps or x > THREE_LINE_RIGHT_X + eps
    else:
        region = "arc_area"
        distance_to_center = math.hypot(x - cx, y - cy)
        on_line = abs(distance_to_center - THREE_LINE_RADIUS) <= eps

        if include_line:
            outside = distance_to_center >= THREE_LINE_RADIUS - eps
        else:
            outside = distance_to_center > THREE_LINE_RADIUS + eps

    return {
        "outside": bool(outside),
        "on_line": bool(on_line),
        "region": region,
        "distance_to_center": distance_to_center,
    }


def load_detection_json(json_path: str) -> Dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict[str, Any], output_path: str) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return str(output_file)


def default_score_output_path(json_path: str) -> str:
    path = Path(json_path)
    return str(path.with_name(f"{path.stem}_score.json"))


def default_visualization_output_path(image_path: str, json_path: str) -> str:
    if image_path:
        path = Path(image_path)
        return str(path.with_name(f"{path.stem}_score_vis{path.suffix}"))

    path = Path(json_path)
    return str(path.with_name(f"{path.stem}_score_vis.png"))


def resolve_image_path(image_path: Optional[str], json_path: str) -> Optional[str]:
    if not image_path:
        return None

    path = Path(image_path)
    if path.is_absolute() or path.exists():
        return str(path)

    json_relative_path = Path(json_path).parent / path
    if json_relative_path.exists():
        return str(json_relative_path)

    return str(path)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_valid_point(x: float, y: float) -> bool:
    return not (x == 0.0 and y == 0.0)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _keypoints_by_label(keypoints: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(keypoint.get("label", "")).upper(): keypoint
        for keypoint in keypoints
        if isinstance(keypoint, dict)
    }


def collect_homography_points(
    detections: Dict[str, Any],
    conf_threshold: float = DEFAULT_KEYPOINT_CONF_THRESHOLD,
) -> Tuple[List[Point], List[Point], List[Dict[str, Any]]]:
    """
    从 JSON 中提取参与单应性矩阵计算的点。

    低于 conf_threshold 的关键点会被过滤掉。
    """
    keypoints = detections.get("keypoints", [])
    by_label = _keypoints_by_label(keypoints)

    src_pts: List[Point] = []
    dst_pts: List[Point] = []
    valid_keypoints: List[Dict[str, Any]] = []

    for index, label in enumerate(KEYPOINT_LABELS):
        keypoint = by_label.get(label)
        if keypoint is None and index < len(keypoints) and isinstance(keypoints[index], dict):
            keypoint = keypoints[index]

        if not keypoint:
            continue

        x = _to_float(keypoint.get("x"))
        y = _to_float(keypoint.get("y"))
        conf = _to_float(keypoint.get("conf"))

        if conf < conf_threshold or not _is_valid_point(x, y):
            continue

        template_point = TEMPLATE_KEYPOINTS[index]
        src_pts.append((x, y))
        dst_pts.append(template_point)
        valid_keypoints.append(
            {
                "label": label,
                "source": [x, y],
                "template": list(template_point),
                "conf": conf,
            }
        )

    return src_pts, dst_pts, valid_keypoints


def person_foot_point(person: PersonDetection) -> Optional[Point]:
    """
    人物脚点取人物框底边中心点：((x1 + x2) / 2, y2)。
    """
    xyxy = person.get("xyxy")
    if not isinstance(xyxy, SequenceABC) or len(xyxy) < 4:
        return None

    x1, _, x2, y2 = (_to_float(value) for value in xyxy[:4])
    return (x1 + x2) / 2.0, y2


def _default_person_score(
    person: PersonDetection,
    reason: str,
    foot_point: Optional[Point] = None,
) -> Dict[str, Any]:
    return {
        "id": person.get("id"),
        "conf": _to_float(person.get("conf")),
        "xyxy": person.get("xyxy", []),
        "foot_point": list(foot_point) if foot_point is not None else None,
        "template_foot_point": None,
        "three_line_status": None,
        "outside_three_line": False,
        "score": DEFAULT_SCORE_WITHOUT_HOMOGRAPHY,
        "reason": reason,
    }


def score_person(
    person: PersonDetection,
    homography: Optional[np.ndarray],
    include_line: bool = True,
    eps: float = THREE_LINE_EPS,
    default_reason: str = "not_enough_keypoints_default_two_points",
) -> Dict[str, Any]:
    foot_point = person_foot_point(person)

    if homography is None:
        return _default_person_score(
            person,
            default_reason,
            foot_point=foot_point,
        )

    if foot_point is None:
        return _default_person_score(person, "invalid_person_box_default_two_points")

    try:
        template_foot_point = map_point_A_to_warpedA(foot_point, homography)
        status = compute_three_point_status(
            template_foot_point,
            include_line=include_line,
            eps=eps,
        )
    except ValueError as error:
        return _default_person_score(person, str(error), foot_point=foot_point)

    outside = bool(status["outside"])
    return {
        "id": person.get("id"),
        "conf": _to_float(person.get("conf")),
        "xyxy": person.get("xyxy", []),
        "foot_point": list(foot_point),
        "template_foot_point": [template_foot_point[0], template_foot_point[1]],
        "three_line_status": status,
        "outside_three_line": outside,
        "score": 3 if outside else 2,
        "reason": "ok",
    }


def _person_color(person: PersonDetection) -> Tuple[int, int, int]:
    if person.get("reason") != "ok":
        return COLOR_DEFAULT_SCORE
    return COLOR_THREE_POINT if int(person.get("score", 2)) == 3 else COLOR_TWO_POINT


def _score_label(person: PersonDetection) -> str:
    person_id = person.get("id", "?")
    score = int(person.get("score", DEFAULT_SCORE_WITHOUT_HOMOGRAPHY))

    if person.get("reason") != "ok":
        area = "DEFAULT"
    elif person.get("outside_three_line"):
        area = "OUT"
    else:
        area = "IN"

    return f"ID {person_id} | {score}PT | {area}"


def _safe_xyxy(image: np.ndarray, xyxy: Sequence[Any]) -> Optional[Tuple[int, int, int, int]]:
    if len(xyxy) < 4:
        return None

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(_to_float(value))) for value in xyxy[:4]]
    return (
        _clamp(x1, 0, image_width - 1),
        _clamp(y1, 0, image_height - 1),
        _clamp(x2, 0, image_width - 1),
        _clamp(y2, 0, image_height - 1),
    )


def _safe_point(image: np.ndarray, point: Optional[Sequence[Any]]) -> Optional[Tuple[int, int]]:
    if point is None or len(point) < 2:
        return None

    image_height, image_width = image.shape[:2]
    x = _clamp(int(round(_to_float(point[0]))), 0, image_width - 1)
    y = _clamp(int(round(_to_float(point[1]))), 0, image_height - 1)
    return x, y


def _draw_label(
    cv2_module: Any,
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2_module.getTextSize(
        text,
        font,
        FONT_SCALE,
        FONT_THICKNESS,
    )

    image_height, image_width = image.shape[:2]
    x = _clamp(
        origin[0],
        LABEL_PADDING,
        max(LABEL_PADDING, image_width - text_width - LABEL_PADDING),
    )
    y = _clamp(
        origin[1],
        text_height + LABEL_PADDING,
        max(text_height + LABEL_PADDING, image_height - baseline - LABEL_PADDING),
    )

    top_left = (x - LABEL_PADDING, y - text_height - LABEL_PADDING)
    bottom_right = (x + text_width + LABEL_PADDING, y + baseline + LABEL_PADDING)
    cv2_module.rectangle(image, top_left, bottom_right, COLOR_TEXT_BG, -1)
    cv2_module.rectangle(image, top_left, bottom_right, color, 2)
    cv2_module.putText(
        image,
        text,
        (x, y),
        font,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv2_module.LINE_AA,
    )


def draw_score_report_on_image(
    report: Dict[str, Any],
    image_path: str,
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    将每个人物的 2 分/3 分结果直接绘制到原图上。
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"image not found: {image_path}")

    for person in report.get("persons", []):
        if not isinstance(person, dict):
            continue

        color = _person_color(person)
        label = _score_label(person)

        xyxy = person.get("xyxy", [])
        box = _safe_xyxy(image, xyxy) if isinstance(xyxy, SequenceABC) else None
        foot_point = _safe_point(image, person.get("foot_point"))

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            label_origin = (x1, y1 - 8)
        elif foot_point is not None:
            label_origin = (foot_point[0] + 10, foot_point[1] - 10)
        else:
            continue

        if foot_point is not None:
            cv2.circle(image, foot_point, 7, color, -1)
            cv2.circle(image, foot_point, 7, COLOR_TEXT_BG, 2)

        _draw_label(cv2, image, label, label_origin, color)

    if output_path is None:
        image_file = Path(image_path)
        output_path = str(
            image_file.with_name(f"{image_file.stem}_score_vis{image_file.suffix}")
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_file), image):
        raise IOError(f"failed to write image: {output_file}")

    if show:
        cv2.imshow("score_visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return str(output_file)


def build_score_report(
    detections: Dict[str, Any],
    conf_threshold: float = DEFAULT_KEYPOINT_CONF_THRESHOLD,
    include_line: bool = True,
    eps: float = THREE_LINE_EPS,
) -> Dict[str, Any]:
    src_pts, dst_pts, valid_keypoints = collect_homography_points(
        detections,
        conf_threshold=conf_threshold,
    )

    homography = None
    homography_error = None
    if len(src_pts) >= MIN_HOMOGRAPHY_POINTS:
        try:
            homography = find_homography(src_pts, dst_pts)
        except (ValueError, np.linalg.LinAlgError) as error:
            homography_error = str(error)
    else:
        homography_error = "not_enough_keypoints_default_two_points"

    persons = detections.get("persons", [])
    scored_persons = [
        score_person(
            person,
            homography,
            include_line=include_line,
            eps=eps,
            default_reason=homography_error or "homography_unavailable_default_two_points",
        )
        for person in persons
        if isinstance(person, dict)
    ]

    return {
        "image_path": detections.get("image_path"),
        "image_width": detections.get("image_width"),
        "image_height": detections.get("image_height"),
        "visualization_path": detections.get("visualization_path"),
        "keypoint_conf_threshold": conf_threshold,
        "valid_keypoint_count": len(valid_keypoints),
        "valid_keypoints": valid_keypoints,
        "homography_available": homography is not None,
        "homography_error": homography_error,
        "homography": homography.tolist() if homography is not None else None,
        "persons": scored_persons,
    }


def score_from_detection_json(
    json_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = DEFAULT_KEYPOINT_CONF_THRESHOLD,
    include_line: bool = True,
    eps: float = THREE_LINE_EPS,
    save_result: bool = True,
    visualize: bool = True,
    visualization_output_path: Optional[str] = None,
    image_path: Optional[str] = None,
    show_visualization: bool = False,
) -> Dict[str, Any]:
    """
    读取检测 JSON，计算人物脚点的模板坐标，并判断 2 分 / 3 分。

    如果有效关键点少于 4 个，所有人物默认按 2 分处理。
    """
    detections = load_detection_json(json_path)
    report = build_score_report(
        detections,
        conf_threshold=conf_threshold,
        include_line=include_line,
        eps=eps,
    )
    report["source_json_path"] = str(json_path)

    resolved_image_path = resolve_image_path(image_path or report.get("visualization_path"), json_path)
    if visualize and resolved_image_path:
        visualization_output_path = visualization_output_path or default_visualization_output_path(
            resolved_image_path,
            json_path,
        )
        report["score_visualization_path"] = draw_score_report_on_image(
            report,
            resolved_image_path,
            output_path=visualization_output_path,
            show=show_visualization,
        )

    if save_result:
        output_path = output_path or default_score_output_path(json_path)
        report["score_json_path"] = save_json(report, output_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge 2/3 point score from detection JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="strategy/test_detections.json",
        help="Path to detection JSON.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save score JSON. Defaults to *_score.json beside input.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_KEYPOINT_CONF_THRESHOLD,
        help="Ignore keypoints whose confidence is lower than this value.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=THREE_LINE_EPS,
        help="Tolerance for judging the three-point line in template coordinates.",
    )
    parser.add_argument(
        "--line-is-inside",
        action="store_true",
        help="Treat points on the three-point line as inside/two-point.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the score JSON file.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Original image path. Defaults to image_path in detection JSON.",
    )
    parser.add_argument(
        "--vis-output",
        default=None,
        help="Path to save visualization image. Defaults to *_score_vis beside original image.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not draw scores on the original image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization window after saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = score_from_detection_json(
        args.json_path,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        include_line=not args.line_is_inside,
        eps=args.eps,
        save_result=not args.no_save,
        visualize=not args.no_visualize,
        visualization_output_path=args.vis_output,
        image_path=args.image,
        show_visualization=args.show,
    )
    print("homography_available:", result["homography_available"])
    print("valid_keypoint_count:", result["valid_keypoint_count"])
    print("score_json_path:", result.get("score_json_path"))
    print("score_visualization_path:", result.get("score_visualization_path"))
    for person in result["persons"]:
        print(
            f"person {person['id']}: score={person['score']}, "
            f"outside_three_line={person['outside_three_line']}, "
            f"template_foot_point={person['template_foot_point']}"
        )
