/*
实现功能：篮球场地线条映射器
初始化：25.08.12 liaoxq
更新：25.09.16 liaoxq
主要功能：
1. 篮球场地线条映射：basketballLineMappingRun
2. 单应性矩阵计算：getHomographyMatrix
3. 线条映射：mapLine, mapAllLines
4. 支持高中和标准两种场地模式
*/

#ifndef BASKETBALLCOURTMAPPER_H
#define BASKETBALLCOURTMAPPER_H

#include "perspectiveTransform.h"
#include <iostream>
#include <vector>
#include <string>

namespace XbotGoAiAnalysis {

// 常量定义
constexpr int REQUIRED_KEY_POINTS_COUNT = 6;  // 用户图像打点数量，必须为6个，且按顺序

// 模板图像尺寸常量
constexpr int HIGHSCHOOL_WIDTH = 1732;
constexpr int HIGHSCHOOL_HEIGHT = 1148;
constexpr int STANDARD_WIDTH = 581;
constexpr int STANDARD_HEIGHT = 384;

// 错误代码枚举
enum class ErrorCode {
    SUCCESS = 0,                    // 成功
    INVALID_KEY_POINTS_COUNT = 1001, // 关键点数量错误
    INSUFFICIENT_POINTS = 1002,     // 点数不足
    INVALID_COORDINATE_RANGE = 1003, // 坐标范围错误[0-1]范围
    INVALID_KEY_POINTS_ORDER = 1004, // 关键点顺序错误
    INVALID_KEY_POINTS_ORDER_1_2 = 10041, // 关键点顺序错误 1-2
    INVALID_KEY_POINTS_ORDER_2_3 = 10042, // 关键点顺序错误 2-3
    INVALID_KEY_POINTS_ORDER_3_4 = 10043, // 关键点顺序错误 3-4
    INVALID_KEY_POINTS_ORDER_5_6 = 10044, // 关键点顺序错误 5-6
    KEY_POINTS_VALIDATION_FAILED = 1005, // 关键点验证失败
    HOMOGRAPHY_CALCULATION_FAILED = 2001, // 单应性矩阵计算失败
    HOMOGRAPHY_MATRIX_INVALID = 2002,     // 单应性矩阵无效
    HOMOGRAPHY_MATRIX_SINGULAR = 2003,    // 单应性矩阵奇异
    POINT_MAPPING_FAILED = 3001,          // 点映射失败
    INVALID_MAPPING_RESULT = 3002,        // 映射结果无效
    UNKNOWN_LINE_TYPE = 4001,             // 未知线条类型
    SVD_DECOMPOSITION_FAILED = 5001,      // SVD分解失败
    MATRIX_DIMENSION_MISMATCH = 5002,     // 矩阵维度不匹配
    MATRIX_TOO_LARGE = 5003,              // 矩阵过大
    MATRIX_EMPTY = 5004,                  // 矩阵为空
    UNKNOWN_ERROR = 9999                  // 未知错误
};

// 错误消息结构
struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::string description;
};

// 语言枚举
enum class Language {
    CHINESE_SIMPLIFIED,
    ENGLISH
};


// 输入参数配置结构体
struct MappingConfig {
    bool isHighschoolMode = false; //默认standard模式
    float imageBWidth = 1920.0;
    float imageBHeight = 1080.0;
    float angleYaw = 0.0;
    float anglePitch = 0.0;
    std::vector<Point2f> pointsBnorm = {};
};

// 定义篮球场地线条类型枚举
enum class LineType {
    SIDE_LINE_1,        // 边线1（顺时针枚举）
    END_LINE_2,         // 底线2
    SIDE_LINE_3,        // 边线3
    THREE_POINT_LINE_4, // 三分线4
    LANE_LINE_5,        // 罚球占位线5
    LANE_LINE_6,        // 罚球占位线6
    FREE_THROW_LINE_7,  // 罚球线7
    FREE_THROW_CIRCLE_8, // 罚球圈8
    FIELD_FILTER_LINES  // 场地过滤线
};

// 单个线条数据结构
struct LineData {
    std::string name;           // 线条名称
    int tag;                    // 线条标签
    std::vector<Point2f> points; // 线条上的点
};

// 半场线条数据结构（保持向后兼容）
struct HalfCourtLines {
    float angleYaw = 0.0;
    float anglePitch = 0.0;
    std::string halfCourtName; // 半场名称: 左半场/右半场
    std::vector<LineData> lines; // 所有线条的集合
};

// 输出结果结构体
struct MappingResult {
    bool success = false;
    bool isHighschoolMode = false; //默认standard模式
    HalfCourtLines halfCourtLines;
    ErrorCode errorCode = ErrorCode::SUCCESS;
    std::string errorMessage = "";
};

// 源线条数据结构（用于存储模板图的线条数据）
struct SourceBasketballLines {
    std::vector<Point2f> sideLine1;        // 边线1
    std::vector<Point2f> endLine2;         // 底线2
    std::vector<Point2f> sideLine3;        // 边线3
    std::vector<Point2f> threePointLine4;  // 三分线4
    std::vector<Point2f> laneLine5;        // 罚球占位线5
    std::vector<Point2f> laneLine6;        // 罚球占位线6
    std::vector<Point2f> freeThrowLine7;   // 罚球线7
    std::vector<Point2f> freeThrowCircle8; // 罚球圈8
    std::vector<Point2f> fieldFilterLines; // 场地过滤线
};


// 定义篮球场地关键点数据结构: A图
struct BasketballKeyPoints {
    std::vector<Point2f> highschoolPoints;
    std::vector<Point2f> standardPoints;
};

// 篮球场映射器类
class BasketballCourtMapper {
public:
    // 构造函数
    BasketballCourtMapper();
    ~BasketballCourtMapper() = default;

    // 核心方法接口：篮球场全场线映射主函数
    MappingResult basketballLineMappingRun(const MappingConfig& config);

    // 获取单应性矩阵
    std::vector<std::vector<double>> getHomographyMatrix() const;

    // 坐标转换方法（归一化 <-> 像素）
    static Point2f pixelToNormalized(const Point2i& pixelPt, int width, int height);
    static Point2i normalizedToPixel(const Point2f& normPt, int width, int height);
    static std::vector<Point2f> pixelToNormalized(const std::vector<Point2i>& pixelPts, int width, int height);
    static std::vector<Point2i> normalizedToPixel(const std::vector<Point2f>& normPts, int width, int height);

    // 错误处理相关方法
    static ErrorInfo getErrorInfo(ErrorCode code, Language lang = Language::CHINESE_SIMPLIFIED);
    static std::string getErrorMessage(ErrorCode code, Language lang = Language::CHINESE_SIMPLIFIED);

private:
    ErrorCode m_errorCode;
    bool m_isHighschool;
    BasketballKeyPoints m_sourceAllKeyPoints;
    SourceBasketballLines m_sourceHighschoolLines;
    SourceBasketballLines m_sourceStandardLines;
    std::vector<Point2f> m_sourceKeyPoints;
    std::vector<Point2f> m_targetKeyPoints;
    std::vector<std::vector<double>> m_homographyMatrix;
    SourceBasketballLines m_sourceLines;

    void initKeyPoints();
    SourceBasketballLines initHighschoolLines();
    SourceBasketballLines initStandardLines();
    void setSourceKeyPoints();
    void setSourceLines();

    // 设置模式（高中或标准场地）
    void setMode(bool isHighschool);

    // 设置目标图像关键点（包含点位校验）
    int validateXOrder(const std::vector<Point2f>& values);
    int validateYOrder(const std::vector<Point2f>& values, float angleYaw);
    bool validateKeyPoints(const std::vector<Point2f>& points, float angleYaw);
    int setTargetKeyPoints(const std::vector<Point2f>& points, float angleYaw);

    // 计算透视变换矩阵
    int calculateHomography();
    double calculateDeterminant3x3(const std::vector<std::vector<double>>& matrix);

    // 映射线条到目标图像
    std::vector<Point2f> mapLine(const std::vector<Point2f>& sourceLine, LineType lineType);

    // 映射特定类型的线条
    std::vector<Point2f> mapLineByType(LineType lineType);

    // 映射所有线条
    std::vector<LineData> mapAllLines();

    // 创建单个线条
    LineData createBasketballLine(LineType lineType, const std::string& name, int tag);

};

} // namespace XbotGoAiAnalysis

#endif // BASKETBALLCOURTMAPPER_H
