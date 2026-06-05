import json
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


KEYPOINT_LABELS = ("A", "B", "C", "D", "E", "F")
KEYPOINT_COUNT = len(KEYPOINT_LABELS)
PERSON_CLASS_ID = 0

DEFAULT_POSE_MODEL_PATH = "model/three-point-line-pose.pt"
DEFAULT_PERSON_MODEL_PATH = "model/yolov11n_1280_person_ball_backboard_hoop_rect.pt"

KEYPOINT_COLOR = (255, 0, 0)  # BGR: blue
PERSON_COLOR = (0, 255, 0)  # BGR: green
TEXT_BACKGROUND_COLOR = (255, 255, 255)

POINT_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_PADDING = 3
LABEL_OFFSET = 10

Keypoint = List[float]
PersonDetection = Dict[str, object]
Color = Tuple[int, int, int]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_at(values: Any, index: int, default: float = 0.0) -> float:
    try:
        return _to_float(values[index], default)
    except (IndexError, TypeError):
        return default


def _empty_keypoints() -> List[Keypoint]:
    return [[0.0, 0.0, 0.0] for _ in range(KEYPOINT_COUNT)]


def _to_numpy(value: Any) -> np.ndarray:
    """兼容 torch tensor / ultralytics 对象 / numpy 数组。"""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _first_row(value: Optional[Any]) -> Optional[np.ndarray]:
    if value is None:
        return None

    array = _to_numpy(value)
    if array.size == 0:
        return None

    return np.atleast_2d(array)[0]


def _keypoint_at(
    coords: Sequence[Sequence[float]],
    confs: Optional[Sequence[float]],
    index: int,
) -> Keypoint:
    if index >= len(coords):
        return [0.0, 0.0, 0.0]

    point = coords[index]
    confidence = confs[index] if confs is not None and index < len(confs) else 0.0
    return [_float_at(point, 0), _float_at(point, 1), _to_float(confidence)]


def extract_keypoints_from_results(results: Iterable[Any]) -> List[Keypoint]:
    """从 ultralytics 推理结果中提取 A-F 六个关键点。

    返回形如 [[x, y, conf], ...] 的列表，长度固定为 6。
    缺失的点会用 [0.0, 0.0, 0.0] 填充。
    """
    for result in results:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            continue

        coords = _first_row(getattr(keypoints, "xy", None))
        if coords is None:
            return _empty_keypoints()

        confs = _first_row(getattr(keypoints, "conf", None))
        return [_keypoint_at(coords, confs, index) for index in range(KEYPOINT_COUNT)]

    return _empty_keypoints()


@lru_cache(maxsize=4)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _predict(model_path: str, image_path: str, conf: float, **kwargs: Any) -> Any:
    model = _load_model(str(model_path))
    return model.predict(source=str(image_path), save=False, conf=conf, **kwargs)


def predict_and_extract(
    image_path: str,
    model_path: str = DEFAULT_POSE_MODEL_PATH,
    conf: float = 0.5,
) -> List[Keypoint]:
    """加载姿态模型，对单张图像进行预测，并返回 A-F 关键点。"""
    results = _predict(model_path, image_path, conf, line_thickness=2)
    return extract_keypoints_from_results(results)


def _array_or_default(value: Optional[Any], length: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(length, default)

    array = _to_numpy(value)
    if array.size == 0:
        return np.full(length, default)

    return np.ravel(array)


def extract_persons_from_results(
    results: Iterable[Any],
    person_class_id: int = PERSON_CLASS_ID,
) -> List[PersonDetection]:
    """从 YOLO 检测结果中提取人物框。"""
    persons: List[PersonDetection] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            continue

        xyxy = np.atleast_2d(_to_numpy(boxes.xyxy))
        if xyxy.size == 0:
            continue

        classes = _array_or_default(getattr(boxes, "cls", None), len(xyxy), -1.0)
        confs = _array_or_default(getattr(boxes, "conf", None), len(xyxy), 0.0)

        for box, class_id, score in zip(xyxy, classes, confs):
            if int(class_id) != person_class_id:
                continue

            persons.append(
                {
                    "xyxy": [float(value) for value in box[:4]],
                    "conf": float(score),
                }
            )

    return persons


def predict_persons(
    image_path: str,
    person_model_path: str = DEFAULT_PERSON_MODEL_PATH,
    conf: float = 0.5,
) -> List[PersonDetection]:
    """使用 YOLO 检测图片中的人物。

    返回列表，每项为 {"xyxy": [x1, y1, x2, y2], "conf": float}。
    """
    results = _predict(person_model_path, image_path, conf)
    return extract_persons_from_results(results)


def _default_json_output_path(image_path: str) -> str:
    path = Path(image_path)
    return str(path.with_name(f"{path.stem}_detections.json"))


def _keypoint_to_json(label: str, keypoint: Sequence[float]) -> Dict[str, object]:
    return {
        "label": label,
        "x": _float_at(keypoint, 0),
        "y": _float_at(keypoint, 1),
        "conf": _float_at(keypoint, 2),
    }


def _person_to_json(person: PersonDetection, index: int) -> Dict[str, object]:
    xyxy = person.get("xyxy", [])
    return {
        "id": index + 1,
        "xyxy": [
            _float_at(xyxy, 0),
            _float_at(xyxy, 1),
            _float_at(xyxy, 2),
            _float_at(xyxy, 3),
        ],
        "conf": _to_float(person.get("conf", 0.0)),
    }


def save_detection_result_json(
    image_path: str,
    keypoints_list: Sequence[Sequence[float]],
    persons: Sequence[PersonDetection],
    output_path: Optional[str] = None,
    visualization_path: Optional[str] = None,
) -> str:
    """保存关键点和人物框信息到 JSON 文件，返回 JSON 路径。"""
    #读取图片 获取 wh
    img = _read_image(image_path)
    keypoints = list(keypoints_list)
    output_path = output_path or _default_json_output_path(image_path)

    data = {
        "image_path": str(image_path),
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "visualization_path": visualization_path,
        "keypoints": [
            _keypoint_to_json(label, keypoints[index] if index < len(keypoints) else [])
            for index, label in enumerate(KEYPOINT_LABELS)
        ],
        "persons": [_person_to_json(person, index) for index, person in enumerate(persons)],
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return str(output_file)


def _read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    return image


def _default_output_path(image_path: str, suffix: str) -> str:
    path = Path(image_path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def _save_and_maybe_show(
    image: np.ndarray,
    image_path: str,
    output_path: Optional[str],
    suffix: str,
    window_name: str,
    show: bool,
) -> str:
    output_path = output_path or _default_output_path(image_path, suffix)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"failed to write image: {output_path}")

    if show:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return str(output_path)


def _fit_label_origin(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
) -> Tuple[int, int, int, int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
    )

    image_height, image_width = image.shape[:2]
    min_x = LABEL_PADDING
    min_y = text_height + LABEL_PADDING
    max_x = max(min_x, image_width - text_width - LABEL_PADDING)
    max_y = max(min_y, image_height - baseline - LABEL_PADDING)

    x = min(max(origin[0], min_x), max_x)
    y = min(max(origin[1], min_y), max_y)
    return x, y, text_width, text_height, baseline


def _draw_label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Color) -> None:
    x, y, text_width, text_height, baseline = _fit_label_origin(image, text, origin)

    top_left = (x - LABEL_PADDING, y - text_height - LABEL_PADDING)
    bottom_right = (x + text_width + LABEL_PADDING, y + baseline + LABEL_PADDING)
    cv2.rectangle(image, top_left, bottom_right, TEXT_BACKGROUND_COLOR, -1)
    cv2.rectangle(image, top_left, bottom_right, color, 1)
    cv2.putText(image, text, (x, y), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)


def _is_missing_keypoint(keypoint: Sequence[float]) -> bool:
    return len(keypoint) < 3 or (
        _float_at(keypoint, 0) == 0.0
        and _float_at(keypoint, 1) == 0.0
        and _float_at(keypoint, 2) == 0.0
    )


def _keypoint_label_origin(
    image: np.ndarray,
    point: Tuple[int, int],
    text: str,
) -> Tuple[int, int]:
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
    )

    x, y = point
    origin_x = x + LABEL_OFFSET
    origin_y = y - LABEL_OFFSET
    image_height, image_width = image.shape[:2]

    if origin_x + text_width + LABEL_PADDING > image_width:
        origin_x = x - LABEL_OFFSET - text_width
    if origin_y - text_height - LABEL_PADDING < 0:
        origin_y = y + LABEL_OFFSET + text_height

    return origin_x, origin_y


def _draw_keypoint(image: np.ndarray, label: str, keypoint: Sequence[float]) -> None:
    if _is_missing_keypoint(keypoint):
        return

    x, y, confidence = keypoint[:3]
    point = (int(round(float(x))), int(round(float(y))))
    text = f"{label} {float(confidence):.2f}"

    cv2.circle(image, point, POINT_RADIUS, KEYPOINT_COLOR, -1)
    cv2.circle(image, point, POINT_RADIUS, TEXT_BACKGROUND_COLOR, 2)
    _draw_label(image, text, _keypoint_label_origin(image, point, text), KEYPOINT_COLOR)


def _draw_keypoints(image: np.ndarray, keypoints_list: Sequence[Sequence[float]]) -> None:
    for label, keypoint in zip(KEYPOINT_LABELS, keypoints_list):
        _draw_keypoint(image, label, keypoint)


def draw_keypoints_on_image_cv(
    image_path: str,
    keypoints_list: Sequence[Sequence[float]],
    output_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """使用 OpenCV 在图片上按照 A-F 顺序可视化关键点。"""
    image = _read_image(image_path)
    _draw_keypoints(image, keypoints_list)
    return _save_and_maybe_show(image, image_path, output_path, "_kps", "keypoints", show)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _safe_box_xyxy(image: np.ndarray, xyxy: Sequence[float]) -> Tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy[:4]]
    return (
        _clamp(x1, 0, image_width - 1),
        _clamp(y1, 0, image_height - 1),
        _clamp(x2, 0, image_width - 1),
        _clamp(y2, 0, image_height - 1),
    )


def _draw_person(image: np.ndarray, person: PersonDetection, index: int) -> None:
    xyxy = person.get("xyxy")
    try:
        if xyxy is None or len(xyxy) < 4:
            return
    except TypeError:
        return

    x1, y1, x2, y2 = _safe_box_xyxy(image, xyxy)
    confidence = _to_float(person.get("conf", 0.0))
    label = f"Person {index + 1} {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), PERSON_COLOR, 2)
    _draw_label(image, label, (x1, y1 - 6), PERSON_COLOR)


def _draw_persons(image: np.ndarray, persons: Sequence[PersonDetection]) -> None:
    for index, person in enumerate(persons):
        _draw_person(image, person, index)


def draw_persons_and_keypoints(
    image_path: str,
    keypoints_list: Sequence[Sequence[float]],
    persons: Sequence[PersonDetection],
    output_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """在同一张图上绘制人物框和 A-F 点，返回保存路径。"""
    image = _read_image(image_path)
    _draw_persons(image, persons)
    _draw_keypoints(image, keypoints_list)
    return _save_and_maybe_show(
        image,
        image_path,
        output_path,
        "_persons_kps",
        "persons_and_kps",
        show,
    )


def process_and_visualize(
    image_path: str,
    pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
    person_model_path: str = DEFAULT_PERSON_MODEL_PATH,
    conf: float = 0.5,
    show: bool = True,
    verbose: bool = True,
    save_json: bool = True,
    json_output_path: Optional[str] = None,
) -> str:
    """对图片进行关键点姿态预测与人物检测，并可视化两者。"""
    keypoints = predict_and_extract(image_path, model_path=pose_model_path, conf=conf)
    if verbose:
        print("预测的关键点坐标和置信度:")
        pprint(keypoints)

    persons = predict_persons(image_path, person_model_path, conf=conf)
    visualization_path = draw_persons_and_keypoints(image_path, keypoints, persons, show=show)

    if save_json:
        detection_json_path = save_detection_result_json(
            image_path,
            keypoints,
            persons,
            output_path=json_output_path,
            visualization_path=visualization_path,
        )
        if verbose:
            print("Saved detection json to:", detection_json_path)

    return visualization_path


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Generate detection data for a single image.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--pose_model", default=DEFAULT_POSE_MODEL_PATH, help="Path to the pose estimation model.")
    parser.add_argument("--person_model", default=DEFAULT_PERSON_MODEL_PATH, help="Path to the person detection model.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for predictions.")
    parser.add_argument("--show", action="store_true", help="Whether to display the visualization window.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed prediction information.")
    parser.add_argument("--no-json", dest="save_json", action="store_false", help="Whether to save detection results to JSON.")
    parser.add_argument("--json_output", default=None, help="Path to save the detection JSON file (optional).")
    args = parser.parse_args()
    try:
        output = process_and_visualize(
            image_path=args.image,
            pose_model_path=args.pose_model,
            person_model_path=args.person_model,
            conf=args.conf,
            show=args.show,
            verbose=args.verbose,
            save_json=args.save_json,
            json_output_path=args.json_output,
        )
        print("Saved merged visualization to:", output)
    except Exception as error:
        print("Error running process_and_visualize:", error)
