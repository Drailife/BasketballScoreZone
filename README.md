# 篮球AI分析：场地线单应性矩阵和透视变换核心算法

## 算法流程

**输入**: 用户图像6个关键点（归一化坐标）+ 图像尺寸 + 角度信息

**流程**:
1. 计算单应性矩阵（模板点 ↔ 用户点）
2. 透视变换映射（模板线条 → 用户图像线条）

**输出**:
1. JSON格式的场地线条坐标数据
2. 映射错误，返回错误信息

## 代码结构

当前仓库已经整理为“C++ 主流程 + Python 数据生成/可视化 + 测试文件”三部分：

```
BasketballCourtMapper.cpp / BasketballCourtMapper.h   # 场地映射主入口
ScoreStrategy.cpp / ScoreStrategy.h                   # 三分线判断逻辑
perspectiveTransform.cpp / perspectiveTransform.h     # 单应性矩阵和透视变换
main.cpp                                              # C++ 调用示例

test/                                                 # C++ 测试入口
    ├── test_ScoreStrategy.cpp
    └── test_perspectiveTrans.cpp

strategy/                                             # JSON 生成和测试结果
    ├── GenerateDetData.py
    ├── ScoreStrategy.py
    ├── test_detections.json
    └── test_detections_score.json

python/                                               # OpenCV 可视化与辅助脚本
```

## 使用方法

```cpp
#include "BasketballCourtMapper.h"

// 1. 配置输入参数
MappingConfig config;
config.isHighschoolMode = true;  // 高中/标准场地模式
config.imageBWidth = 1920.0;     // 图像宽度
config.imageBHeight = 1080.0;    // 图像高度
config.angleYaw = 50.0;          // 角度信息
config.anglePitch = 0.0;
config.pointsBnorm = userPoints; // 用户打点（6个，归一化坐标）

// 2. 执行映射
BasketballCourtMapper mapper;
MappingResult result = mapper.basketballLineMappingRun(config);

// 3. 获取结果
if (result.success) {
    // 成功：result.halfCourtLines 包含所有场地线条数据
} else {
    // 失败：result.errorMessage 包含错误信息
}
```

### 结构体

```cpp
// 输入：参数配置结构体
struct MappingConfig {
    bool isHighschoolMode = false; //默认standard模式
    float imageBWidth;
    float imageBHeight;
    float angleYaw;
    float anglePitch;
    std::vector<Point2f> pointsBnorm;
};

// 其中，数据类型：
struct Point2f {
    float x, y;
    Point2f() : x(0.0f), y(0.0f) {}
    Point2f(float x, float y) : x(x), y(y) {}
};

// 输出：输出结构体 
struct MappingResult {
    bool success;
    bool isHighschoolMode;
    HalfCourtLines halfCourtLines;
    std::string errorMessage;
};
```

## 输出格式

```json
{
    "yaw": 50,
    "pitch": 0,
    "halfcourt": "rightCourt",
    "lines": [
        {
            "name": "sideLine",
            "tag": 1,
            "points": [{"x": 0.415856, "y": 0.390782}, ...]
        },
        {
            "name": "endLine", 
            "tag": 2,
            "points": [{"x": 0.416826, "y": 0.390694}, ...]
        }
    ]
}
```

## 编译运行

仓库当前使用 CMake 编译：

```bash
cmake -S . -B build
cmake --build build -j
```

运行可执行文件：

```bash
./build/test_score
./build/test_perspectiveTrans
./build/demo
```

如果要清理生成文件：

```bash
rm -rf build
```

## 错误代码表

| 错误代码 | 错误名称 | 中文错误信息 | 英文错误信息 | 中文描述 | 英文描述 |
|---------|---------|-------------|-------------|----------|----------|
| 0 | SUCCESS | 成功 | Success | 操作成功完成 | Operation completed successfully |
| 1001 | INVALID_KEY_POINTS_COUNT | 关键点数量错误 | Invalid key points count | 关键点数量必须为6个 | Key points count must be 6 |
| 1002 | INSUFFICIENT_POINTS | 点数不足 | Insufficient points | 打点点位数量不足2个 | Insufficient marked points (less than 2) |
| 1003 | INVALID_COORDINATE_RANGE | 坐标范围错误 | Invalid coordinate range | 关键点坐标必须在[0,1]范围内 | Key point coordinates must be in [0,1] range |
| 1004 | INVALID_KEY_POINTS_ORDER | 关键点顺序错误 | Invalid key points order | 关键点顺序不符合要求 | Key points order does not meet requirements |
| 10041 | INVALID_KEY_POINTS_ORDER_1_2 | 检测到1号点位和2号点位错误，请进行调整 | Detected 1st and 2nd key points error, please adjust | 检测到1号点位和2号点位错误，请进行调整 | Detected 1st and 2nd key points error, please adjust |
| 10042 | INVALID_KEY_POINTS_ORDER_2_3 | 检测到2号点位和3号点位错误，请进行调整 | Detected 2nd and 3rd key points error, please adjust | 检测到2号点位和3号点位错误，请进行调整 | Detected 2nd and 3rd key points error, please adjust |
| 10043 | INVALID_KEY_POINTS_ORDER_3_4 | 检测到3号点位和4号点位错误，请进行调整 | Detected 3rd and 4th key points error, please adjust | 检测到3号点位和4号点位错误，请进行调整 | Detected 3rd and 4th key points error, please adjust |
| 10044 | INVALID_KEY_POINTS_ORDER_5_6 | 检测到5号点位和6号点位错误，请进行调整 | Detected 5th and 6th key points error, please adjust | 检测到5号点位和6号点位错误，请进行调整 | Detected 5th and 6th key points error, please adjust |
| 1005 | KEY_POINTS_VALIDATION_FAILED | 关键点验证失败 | Key points validation failed | 关键点验证未通过 | Key points validation did not pass |
| 2001 | HOMOGRAPHY_CALCULATION_FAILED | 单应性矩阵计算失败 | Mapping calculation failed | 计算单应性矩阵时出错 | Error occurred while calculating mapping matrix |
| 2002 | HOMOGRAPHY_MATRIX_INVALID | 单应性矩阵无效 | Invalid mapping matrix | 单应性矩阵维度不正确 | Mapping matrix dimension is incorrect |
| 2003 | HOMOGRAPHY_MATRIX_SINGULAR | 单应性矩阵奇异 | Singular mapping matrix | 单应性矩阵接近奇异，计算结果不可靠 | Mapping matrix is near singular, calculation result unreliable |
| 3001 | POINT_MAPPING_FAILED | 点映射失败 | Point mapping failed | 点映射过程中出现错误 | Error occurred during point mapping |
| 3002 | INVALID_MAPPING_RESULT | 映射结果无效 | Invalid mapping result | 点映射结果无效 | Point mapping result is invalid |
| 4001 | UNKNOWN_LINE_TYPE | 未知线条类型 | Unknown line type | 指定的线条类型不存在 | Specified line type does not exist |
| 5001 | SVD_DECOMPOSITION_FAILED | SVD分解失败 | SVD decomposition failed | 矩阵SVD分解过程中出现错误 | Error occurred during matrix SVD decomposition |
| 5002 | MATRIX_DIMENSION_MISMATCH | 矩阵维度不匹配 | Matrix dimension mismatch | 矩阵维度不一致，无法相乘 | Matrix dimensions are inconsistent, cannot multiply |
| 5003 | MATRIX_TOO_LARGE | 矩阵过大 | Matrix too large | 矩阵过大，可能导致计算不稳定 | Matrix is too large, may cause calculation instability |
| 5004 | MATRIX_EMPTY | 矩阵为空 | Matrix is empty | 输入矩阵为空 | Input matrix is empty |
| 9999 | UNKNOWN_ERROR | 未知错误 | Unknown error | 发生了未知错误 | An unknown error occurred |

### 错误代码分类

- **1000-1999**: 输入参数验证错误
- **2000-2999**: 单应性矩阵计算错误
- **3000-3999**: 点映射和结果验证错误
- **4000-4999**: 线条类型相关错误
- **5000-5999**: 数学计算错误
- **9999**: 未知错误

## 技术文档

[详细算法文档](https://doc.weixin.qq.com/doc/w3_AWkA7QaoAF0CNJJaBiKWiSzCLPhhP?scode=ADgAsAdWAA4ZWdFohcAWkA7QaoAF0)




## 三分线判定与可视化

当前仓库新增了 `ScoreStrategy` 的三分线判定流程，以及 `test/test_ScoreStrategy.cpp` 的 JSON 回放测试。

### JSON 生成来源

`strategy/test_detections_score.json` 由 `strategy/GenerateDetData.py` 和 `strategy/ScoreStrategy.py` 生成：
- `strategy/GenerateDetData.py` 负责读取检测结果并整理成统一的 JSON 结构
- `strategy/ScoreStrategy.py` 负责三分线可视化、关键点判断与最终结果输出

如果需要重新生成测试 JSON，请先确保这两个脚本使用的是同一份输入图像和同一套模型输出。

### 输入数据

测试文件使用 `strategy/test_detections_score.json`，其中包含：
- `valid_keypoints`: 三分线 6 个关键点 A-F
- `persons`: 检测到的人物框、脚点和置信度
- `image_width` / `image_height`: 原图尺寸

### 判定逻辑

1. 先读取 A-F 三分线关键点。
2. 构建单应性矩阵。
3. 再读取每个人的脚点。
4. 对每个人的脚点调用 `ScoreStrategy::judgePlayerPoint`。
5. 输出该点位是三分线内还是三分线外，结果为 2 分或 3 分。

### 编译方式

项目根目录使用 CMake：

```bash
cmake -S . -B build
cmake --build build -j
./build/test_score
```

如需运行主程序：

```bash
./build/demo
```

清理生成文件：

```bash
rm -rf build
```

### 注意事项

- `Point2f` 当前结构包含 `x`、`y`、`score` 三个字段。
- 构建单应性矩阵时会自动忽略 `score < 0.5` 的关键点。
- 剔除后若有效点少于 4 个，`ScoreStrategy` 会返回 `false`。

### 目录说明

- `BasketballCourtMapper.cpp/.h`：主映射入口
- `ScoreStrategy.cpp/.h`：三分线判断逻辑
- `perspectiveTransform.cpp/.h`：单应性矩阵和透视变换
- `test/`：测试入口文件
- `strategy/`：JSON 生成脚本、测试数据和可视化结果
- `python/`：Python 可视化与辅助脚本
