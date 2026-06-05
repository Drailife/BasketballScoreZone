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

```
main.cpp  # 调用示例

# 篮球场地数据转换类：模板图，预设场地线、点，归一化数据处理：
BasketballCourtMapper.cpp  
BasketballCourtMapper.h 

# 数学计算核心函数类：单应性矩阵计算，透视变换计算，矩阵运算，无cv：
perspectiveTransform.cpp  
perspectiveTransform.h

# python代码库参考：模板图、cv打点画图、一致性验证：
python/

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

```bash
mkdir build && cd build
cmake ..
cmake --build .
./bin/demoAnalysis  # 运行程序
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


