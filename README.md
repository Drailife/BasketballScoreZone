# BasketballScoreZone

篮球场地映射与球员得分区域判断工具。

项目当前主要包含两条能力：

- 场地线映射：根据球场 A-F 关键点计算单应性矩阵，将模板场地线映射到原图。
- 得分区域判断：读取检测 JSON 中的 `keypoints` 和 `persons`，用人物脚点判断 1 分、2 分或 3 分区域。

核心逻辑在 C++ 中实现，Python 负责检测 JSON 生成、结果回放和图片可视化。

## 核心流程

```text
输入图像
  -> strategy/GenerateDetData.py 生成检测 JSON
  -> C++ ScoreStrategy 读取 A-F 关键点并内部过滤低置信点
  -> 计算 原图 -> 模板图 的单应性矩阵
  -> 取人物框底边中心作为脚点
  -> 将脚点映射到模板坐标
  -> 判断 1 分 / 2 分 / 3 分
  -> Python 可选绘制得分结果，并把模板图叠加到结果图上
```

## 代码结构

```text
BasketballCourtMapper.cpp / BasketballCourtMapper.h   场地线映射入口
ScoreStrategy.cpp / ScoreStrategy.h                   得分区域判断策略
perspectiveTransform.cpp / perspectiveTransform.h     单应性矩阵与点透视变换
json.hpp                                              C++ JSON 解析头文件
main.cpp                                              场地线映射调用示例

test/
  test_ScoreStrategy.cpp                              C++ 得分策略 JSON 回放测试
  test_perspectiveTrans.cpp                           C++ 单应性矩阵测试

strategy/
  GenerateDetData.py                                  生成关键点、人物框检测 JSON
  ScoreStrategy.py                                    Python 计分、可视化、模板叠加
  test_detections.json                                示例检测 JSON
  test_detections_score.json                          示例计分输出

python/
  testImgs/                                           示例图片、检测 JSON、可视化结果
  base/                                               标准球场模板图
```

## 输入 JSON

C++ 测试和策略应该读取原始 `keypoints` 字段，不要把 `valid_keypoints` 当作输入。`valid_keypoints` 是 Python 计分后输出的调试字段。

推荐输入格式如下：

```json
{
  "image_path": "python/testImgs/right.png",
  "image_width": 2719,
  "image_height": 1508,
  "keypoints": [
    {"label": "A", "x": 188.4, "y": 17.5, "conf": 0.91},
    {"label": "B", "x": 931.5, "y": 21.8, "conf": 0.88},
    {"label": "C", "x": 1806.8, "y": 17.5, "conf": 0.84},
    {"label": "D", "x": 2541.0, "y": 8.7, "conf": 0.06},
    {"label": "E", "x": 931.5, "y": 974.3, "conf": 0.93},
    {"label": "F", "x": 1797.4, "y": 978.6, "conf": 0.90}
  ],
  "persons": [
    {
      "id": 1,
      "xyxy": [100.0, 200.0, 180.0, 520.0],
      "conf": 0.86
    }
  ]
}
```

字段说明：

- `image_width` / `image_height`：原图尺寸。
- `keypoints`：球场 A-F 六个关键点，坐标是原图像素坐标。
- `persons[].xyxy`：人物框，格式是 `[x1, y1, x2, y2]`，坐标是原图像素坐标。
- `conf`：检测置信度。

## C++ 得分策略

`ScoreStrategy` 对外只需要先初始化一次 A-F 关键点，之后每个人只传人物框或人物脚点即可。

注意：C++ 接口使用归一化坐标，测试文件会把 JSON 像素坐标除以图像宽高。

```cpp
#include "ScoreStrategy.h"

using namespace XbotGoAiAnalysis;

Point2f normalizedPoint(float x, float y, float w, float h, float score = 0.0f) {
    return Point2f(x / w, y / h, score);
}

std::vector<Point2f> keypoints(6);
keypoints[0] = normalizedPoint(xA, yA, imageWidth, imageHeight, confA); // A
keypoints[1] = normalizedPoint(xB, yB, imageWidth, imageHeight, confB); // B
keypoints[2] = normalizedPoint(xC, yC, imageWidth, imageHeight, confC); // C
keypoints[3] = normalizedPoint(xD, yD, imageWidth, imageHeight, confD); // D
keypoints[4] = normalizedPoint(xE, yE, imageWidth, imageHeight, confE); // E
keypoints[5] = normalizedPoint(xF, yF, imageWidth, imageHeight, confF); // F

ScoreStrategy strategy(keypoints);
if (!strategy.isReady()) {
    // 点不足或单应性矩阵失败时，后续判断会默认返回 2 分。
    std::cout << strategy.lastError() << std::endl;
}

Point2f leftTop = normalizedPoint(x1, y1, imageWidth, imageHeight);
Point2f rightBottom = normalizedPoint(x2, y2, imageWidth, imageHeight);

ScoreResult result = strategy.judgePlayerBox(leftTop, rightBottom);
std::cout << "score=" << result.score << ", reason=" << result.reason << std::endl;
```

如果你已经有了人物脚点，可以直接调用：

```cpp
Point2f footPointNorm(footX / imageWidth, footY / imageHeight);
ScoreResult result = strategy.judgePlayerPoint(footPointNorm);
```

## 关键点过滤规则

`ScoreStrategy` 内部会自己决定哪些 A-F 点参与单应性矩阵计算，外部不需要提前筛选。

- 必须按 A-F 顺序传入 6 个点。
- `(0, 0)` 会被视为缺失点并跳过。
- `0 < score < 0.5` 会被视为低置信点并跳过。
- `score == 0` 表示调用方没有提供置信度，只要坐标合法就会参与计算。
- 有效点少于 4 个时，无法计算单应性矩阵，最终默认按 2 分处理。

C++ 中模板 A-F 点已经内置为相对坐标：

```cpp
[(0.0688468, 0.0115942),
 (0.3425129, 0.0144928),
 (0.6643718, 0.0115942),
 (0.9345955, 0.0057971),
 (0.3425129, 0.6463768),
 (0.6609294, 0.6492754)]
```

## 得分判断规则

人物框会先取底边中心作为脚点：

```text
foot_x = (x1 + x2) / 2
foot_y = y2
```

脚点映射到模板图后，按以下逻辑判断：

- 1 分：脚点位于罚球点附近区域，当前阈值为模板像素距离 `25`。
- 3 分：脚点位于三分线外。
- 2 分：不属于 1 分且不在三分线外。

默认 `lineAsOutside = true`，也就是贴近三分线会按线外处理。失败场景下 `ScoreResult.score` 默认是 `2`，`ScoreResult.reason` 会记录原因。

## Python 数据生成

生成检测 JSON：

```bash
python3 strategy/GenerateDetData.py python/testImgs/right.png \
  --json_output python/testImgs/right_detections.json \
  --verbose
```

依赖模型默认位于：

```text
model/three-point-line-pose.pt
model/yolov11n_1280_person_ball_backboard_hoop_rect.pt
```

## Python 计分与可视化

对检测 JSON 进行计分，并在原图上绘制每个人的分数：

```bash
python3 strategy/ScoreStrategy.py python/testImgs/right_detections.json \
  --image python/testImgs/right.png
```

默认会输出：

- `*_score.json`：计分结果 JSON。
- `*_score_vis.png`：在原图上绘制人物框、脚点和 1PT/2PT/3PT。
- `*_score_vis_template_overlay.png`：把模板图透视叠加到已经绘制分数的图片上。

常用参数：

```bash
python3 strategy/ScoreStrategy.py python/testImgs/right_detections.json \
  --image python/testImgs/right.png \
  --vis-output python/testImgs/right_score_vis.png \
  --overlay-output python/testImgs/right_score_overlay.png \
  --overlay-alpha 0.45
```

如果不需要模板叠加：

```bash
python3 strategy/ScoreStrategy.py python/testImgs/right_detections.json --no-template-overlay
```

## 编译运行

项目使用 CMake 编译，C++ 标准为 C++17。

```bash
cmake -S . -B build
cmake --build build -j
```

如果 `build/` 是从旧目录复制或移动过来的，CMake 可能会提示缓存路径不一致。此时先清理旧缓存再重新配置：

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j
```

运行测试：

```bash
./build/test_score
./build/test_perspectiveTrans
```

运行场地线映射示例：

```bash
./build/demo
```

当前 `test/test_ScoreStrategy.cpp` 默认读取：

```text
python/testImgs/right_detections.json
```

如果要测试自己的图片，请先生成同结构的检测 JSON，或修改测试文件中的 JSON 路径。

## 场地线映射接口

除了得分策略，项目仍保留 `BasketballCourtMapper` 场地线映射能力。

```cpp
#include "BasketballCourtMapper.h"

MappingConfig config;
config.isHighschoolMode = false;
config.imageBWidth = 1920.0f;
config.imageBHeight = 1080.0f;
config.angleYaw = 50.0f;
config.anglePitch = 0.0f;
config.pointsBnorm = userPoints; // 6 个归一化关键点

BasketballCourtMapper mapper;
MappingResult result = mapper.basketballLineMappingRun(config);

if (result.success) {
    // result.halfCourtLines 包含映射后的场地线
} else {
    // result.errorMessage 包含失败原因
}
```

## 坐标约定

- JSON 中的关键点和人物框是像素坐标。
- C++ `ScoreStrategy` 输入是归一化坐标，范围 `[0, 1]`。
- C++ 模板 A-F 点使用归一化坐标。
- Python `strategy/ScoreStrategy.py` 内部使用 `581 x 345` 模板像素坐标进行可视化和模板叠加。

## 依赖

C++：

- CMake 3.16+
- 支持 C++17 的编译器
- 仓库内置 `json.hpp`

Python：

- `numpy`
- `opencv-python`
- `ultralytics`

## 开发备注

- 计分策略的主要调用文件是 `ScoreStrategy.cpp` / `ScoreStrategy.h`。
- `test/test_ScoreStrategy.cpp` 展示了从 JSON 读取 `keypoints` 和 `persons.xyxy` 的完整流程。
- 低置信关键点过滤已经放在 `ScoreStrategy` 内部，调用方只需要传原始 A-F 点。
- 当单应性矩阵不可用时，策略会按默认 2 分返回，避免因为关键点缺失导致流程中断。
