#include "BasketballCourtMapper.h"
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace XbotGoAiAnalysis {

// 构造函数
BasketballCourtMapper::BasketballCourtMapper() : m_errorCode(ErrorCode::SUCCESS) {
    // 初始化模板图的A图关键点和线条采样点数据
    initKeyPoints();
    m_sourceHighschoolLines = initHighschoolLines();
    m_sourceStandardLines = initStandardLines();
}

/**
 * @brief 篮球场地线映射主函数
 * 
 * 根据输入的配置参数（包括模式、关键点、角度等），完成篮球场地线的映射。
 * 主要流程包括：
 * 1. 设置当前模式（高中/标准）
 * 2. 设置目标图像关键点（归一化坐标）
 * 3. 计算单应性矩阵
 * 4. 按类型映射所有场地线条
 * 
 * @param config 输入参数配置，包括模式、关键点、图像尺寸、角度等
 * @return MappingResult 返回映射结果，包括是否成功、映射后的线条、错误信息等
 */
MappingResult BasketballCourtMapper::basketballLineMappingRun(const MappingConfig& config) {
    MappingResult result;    
    result.isHighschoolMode = config.isHighschoolMode;
    result.halfCourtLines = HalfCourtLines();
    result.halfCourtLines.angleYaw = config.angleYaw;
    result.halfCourtLines.anglePitch = config.anglePitch;

    // 标记左右半场
    if (config.angleYaw <= 0) {
        result.halfCourtLines.halfCourtName = "leftCourt";
    } else {
        result.halfCourtLines.halfCourtName = "rightCourt";
    }

    // 设置模式
    setMode(config.isHighschoolMode);

    // 设置目标图像B关键点
    int res = setTargetKeyPoints(config.pointsBnorm, config.angleYaw);
    if (res != 0) {
        result.errorCode = m_errorCode;
        result.errorMessage = getErrorMessage(m_errorCode);
        return result;
    }

    // 计算单应性矩阵
    res = calculateHomography();
    if (res != 0) {
        result.errorCode = m_errorCode;
        result.errorMessage = getErrorMessage(m_errorCode);
        return result;
    }

    // 映射所有线条
    result.halfCourtLines.lines = mapAllLines();

    // 检查映射过程中是否有错误
    if (m_errorCode != ErrorCode::SUCCESS) {
        result.errorCode = m_errorCode;
        result.errorMessage = getErrorMessage(m_errorCode);
        result.success = false;
        return result;
    }

    result.success = true;
    return result;
}

// 设置模式（高中或标准场地）：true-美国高中，false-标准
void BasketballCourtMapper::setMode(bool isHighschool) {
    m_isHighschool = isHighschool;
    setSourceKeyPoints();
    setSourceLines();
}

// 设置用户目标图像B关键点，返回：0-成功，-1-失败
int BasketballCourtMapper::setTargetKeyPoints(const std::vector<Point2f>& points, float angleYaw) {
    if (points.size() != REQUIRED_KEY_POINTS_COUNT) {
        m_errorCode = ErrorCode::INVALID_KEY_POINTS_COUNT;
        return -1;
    }

    // 验证归一化坐标范围
    for (const auto& pt : points) {
        if (pt.x < 0.0f || pt.x > 1.0f || pt.y < 0.0f || pt.y > 1.0f || std::isnan(pt.x) || std::isnan(pt.y) || std::isinf(pt.x) || std::isinf(pt.y)) {
            m_errorCode = ErrorCode::INVALID_COORDINATE_RANGE;
            return -1;
        }
    }

    // 校验关键点顺序：
    if (!validateKeyPoints(points, angleYaw)) {
        return -1;
    } else {
        m_targetKeyPoints = points;
        m_errorCode = ErrorCode::SUCCESS;
        return 0;
    }
}

bool BasketballCourtMapper::validateKeyPoints(const std::vector<Point2f>& points, float angleYaw) {
    // 检查输入点数
    if (points.size() != REQUIRED_KEY_POINTS_COUNT) {
        m_errorCode = ErrorCode::INVALID_KEY_POINTS_COUNT;
        return false;
    }

    // 初始画面yaw=0不允许打点（使用浮点数近似比较）
    if (std::abs(angleYaw) < 1e-6f) {
        m_errorCode = ErrorCode::KEY_POINTS_VALIDATION_FAILED;
        return false;
    }

    // 创建前4个点和后2个点的向量
    std::vector<Point2f> firstFour(points.begin(), points.begin() + 4);
    std::vector<Point2f> lastTwo(points.begin() + 4, points.end());

    // 校验前4个点x方向
    int ret = validateXOrder(firstFour);
    if (ret == -1) {
        m_errorCode = ErrorCode::INSUFFICIENT_POINTS;
        return false;
    } else if (ret != 0 && ret != -1) {
        switch (ret) {
            case 1:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_1_2;
                break;
            case 2:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_2_3;
                break;
            case 3:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_3_4;
                break;
            default:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER;
                break;
        }
        return false;
    }

    // 校验前4个点y方向
    ret = validateYOrder(firstFour, angleYaw);
    if (ret == -1) {
        m_errorCode = ErrorCode::INSUFFICIENT_POINTS;
        return false;
    } else if (ret != 0 && ret != -1) {
        switch (ret) {
            case 1:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_1_2;
                break;
            case 2:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_2_3;
                break;
            case 3:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_3_4;
                break;
            default:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER;
                break;
        }
        return false;
    }

    // 校验后2个点
    ret = validateXOrder(lastTwo);
    if (ret == -1) {
        m_errorCode = ErrorCode::INSUFFICIENT_POINTS;
        return false;
    } else if (ret != 0 && ret != -1) {
        switch (ret) {
            case 1:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_5_6;
                break;
            default:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER;
                break;
        }
        return false;
    }

    ret = validateYOrder(lastTwo, angleYaw);
    if (ret == -1) {
        m_errorCode = ErrorCode::INSUFFICIENT_POINTS;
        return false;
    } else if (ret != 0 && ret != -1) {
        switch (ret) {
            case 1:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER_5_6;
                break;
            default:
                m_errorCode = ErrorCode::INVALID_KEY_POINTS_ORDER;
                break;
        }
        return false;
    }

    m_errorCode = ErrorCode::SUCCESS;
    return true;
}

// x方向按大小顺序校验
int BasketballCourtMapper::validateXOrder(const std::vector<Point2f>& values) {
    if (values.size() < 2) {
        return -1; // 返回-1表示size错误
    }

    // 检查x方向是否严格递增：x1<x2<x3<x4
    for (size_t i = 0; i < values.size() - 1; ++i) {
        if (values[i].x >= values[i + 1].x) {
            return i + 1; // 返回错误点位索引, 从第1个点开始
        }
    }

    return 0; // 返回0表示正常
}

// y方向按大小顺序校验
int BasketballCourtMapper::validateYOrder(const std::vector<Point2f>& values, float angleYaw) {
    if (values.size() < 2) {
        return -1; // 返回-1表示size错误
    }

    if (angleYaw <= 0) {
        // 检查左半场：y方向是否递减含相等：y1>=y2>=y3>=y4
        for (size_t i = 0; i < values.size() - 1; ++i) {
            if (values[i].y < values[i + 1].y) {
                return i + 1; // 返回错误点位索引, 从第1个点开始
            }
        }
    } else {
        // 检查右半场：y方向是否递增含相等：y1<=y2<=y3<=y4
        for (size_t i = 0; i < values.size() - 1; ++i) {
            if (values[i].y > values[i + 1].y) {
                return i + 1; // 返回错误点位索引, 从第1个点开始
            }
        }
    }

    return 0; // 返回0表示正常
}


// 计算透视变换矩阵，返回：0-成功，-1-失败
int BasketballCourtMapper::calculateHomography() {
    // 检查关键点数量：必须等于REQUIRED_KEY_POINTS_COUNT（6个点）
    if (m_sourceKeyPoints.size() != REQUIRED_KEY_POINTS_COUNT ||
        m_targetKeyPoints.size() != REQUIRED_KEY_POINTS_COUNT ||
        m_sourceKeyPoints.size() != m_targetKeyPoints.size()) {
        m_errorCode = ErrorCode::INVALID_KEY_POINTS_COUNT;
        return -1;
    }

    try {
        m_homographyMatrix = PerspectiveTransform::findHomography(m_sourceKeyPoints, m_targetKeyPoints);
        
        // 验证单应性矩阵的有效性
        if (m_homographyMatrix.size() != 3) {
            m_errorCode = ErrorCode::HOMOGRAPHY_MATRIX_INVALID;
            return -1;
        }
        // 检查所有行的维度是否一致
        for (const auto& row : m_homographyMatrix) {
            if (row.size() != 3) {
                m_errorCode = ErrorCode::HOMOGRAPHY_MATRIX_INVALID;
                return -1;
            }
        }
        
        // 检查矩阵的行列式是否接近零
        double det = calculateDeterminant3x3(m_homographyMatrix);
        if (det == -1.0) {
            // calculateDeterminant3x3已经设置了错误码
            return -1;
        }
        if (std::abs(det) < 1e-10) {
            m_errorCode = ErrorCode::HOMOGRAPHY_MATRIX_SINGULAR;
            return -1;
        }
        
        m_errorCode = ErrorCode::SUCCESS;
        return 0;
    } catch (const std::exception& e) {
        std::string errorMsg = e.what();
        if (errorMsg == "MATRIX_EMPTY") {
            m_errorCode = ErrorCode::MATRIX_EMPTY;
        } else if (errorMsg == "INVALID_KEY_POINTS_COUNT") {
            m_errorCode = ErrorCode::INVALID_KEY_POINTS_COUNT;
        } else if (errorMsg == "INVALID_COORDINATE_RANGE") {
            m_errorCode = ErrorCode::INVALID_COORDINATE_RANGE;
        } else if (errorMsg == "SVD_DECOMPOSITION_FAILED") {
            m_errorCode = ErrorCode::SVD_DECOMPOSITION_FAILED;
        } else if (errorMsg == "HOMOGRAPHY_MATRIX_SINGULAR") {
            m_errorCode = ErrorCode::HOMOGRAPHY_MATRIX_SINGULAR;
        } else if (errorMsg == "MATRIX_DIMENSION_MISMATCH") {
            m_errorCode = ErrorCode::MATRIX_DIMENSION_MISMATCH;
        } else if (errorMsg == "MATRIX_TOO_LARGE") {
            m_errorCode = ErrorCode::MATRIX_TOO_LARGE;
        } else {
            m_errorCode = ErrorCode::HOMOGRAPHY_CALCULATION_FAILED;
        }
        return -1;
    }
}

// 获取单应性矩阵
std::vector<std::vector<double>> BasketballCourtMapper::getHomographyMatrix() const {
    return m_homographyMatrix;
}

// 映射线条到目标图像: 计算函数
std::vector<Point2f> BasketballCourtMapper::mapLine(const std::vector<Point2f>& sourceLine, LineType lineType) {
    std::vector<Point2f> mappedLine;

    // 检查单应性矩阵是否有效（防御性编程：虽然calculateHomography已经验证过，但这里再次检查以确保安全）
    if (m_homographyMatrix.empty() ||
        m_homographyMatrix.size() != 3) {
        m_errorCode = ErrorCode::HOMOGRAPHY_CALCULATION_FAILED;
        return mappedLine;
    }
    // 检查矩阵第一行的维度（如果矩阵结构不正确，后续访问会出错）
    if (m_homographyMatrix[0].size() != 3) {
        m_errorCode = ErrorCode::HOMOGRAPHY_MATRIX_INVALID;
        return mappedLine;
    }

    // 如果源线条为空，直接返回空向量（不设置错误码，因为空线条可能是合法的）
    if (sourceLine.empty()) {
        return mappedLine;
    }

    // 预分配内存以提高性能
    mappedLine.reserve(sourceLine.size());

    // 记录是否有错误发生，但不立即设置错误码（避免覆盖）
    bool hasInvalidMapping = false;
    bool hasOutOfRange = false;
    int successCount = 0;

    for (const auto& pt : sourceLine) {
        try {
            Point2f mappedPt = PerspectiveTransform::mapPointAToWarpedA(pt, m_homographyMatrix);
            
            // 检查映射结果是否有效（NaN和Inf检查）
            if (std::isnan(mappedPt.x) || std::isnan(mappedPt.y) || 
                std::isinf(mappedPt.x) || std::isinf(mappedPt.y)) {
                hasInvalidMapping = true;
                // 跳过无效点，继续处理其他点
                continue;
            }

            // 检查坐标是否在画面范围内（0-1）
            // 对于场地过滤线，保留所有点，包括超出画面范围的点
            if (lineType != LineType::FIELD_FILTER_LINES && 
                (mappedPt.x < 0.0f || mappedPt.x > 1.0f ||
                 mappedPt.y < 0.0f || mappedPt.y > 1.0f)) {
                hasOutOfRange = true;
                // 跳过超出画面的点，继续处理其他点
                continue;
            }

            if (lineType == LineType::FIELD_FILTER_LINES &&
                (mappedPt.x < -3.0f || mappedPt.x > 3.0f ||
                 mappedPt.y < -3.0f || mappedPt.y > 3.0f)) {
                hasOutOfRange = true;
                // FIELD_FILTER_LINES类型跳过超出画面3倍距离的点，继续处理其他点
                continue;
            }

            mappedLine.push_back(mappedPt);
            successCount++;
        } catch (const std::exception& e) {
            // 跳过异常点，继续处理其他点
            // 注意：不在这里设置错误码，因为部分点失败不应该阻止整个映射过程
            hasInvalidMapping = true;
            continue;
        }
    }

    // 根据映射结果设置错误码（优先级：全部失败 > 无效映射 > 超出范围 > 成功）
    if (mappedLine.empty() && !sourceLine.empty()) {
        m_errorCode = ErrorCode::POINT_MAPPING_FAILED;
    } else if (hasInvalidMapping && successCount == 0) {
        m_errorCode = ErrorCode::INVALID_MAPPING_RESULT;
    } else if (hasOutOfRange && successCount == 0) {
        m_errorCode = ErrorCode::INVALID_COORDINATE_RANGE;
    } else {
        // 部分点成功，不设置错误码（保持之前的错误码或SUCCESS）
        if (successCount > 0) {
            m_errorCode = ErrorCode::SUCCESS;
        }
    }
    return mappedLine;
}

// 映射特定类型的线条：选择单一线条类型，单条线映射计算
std::vector<Point2f> BasketballCourtMapper::mapLineByType(LineType lineType) {
    std::vector<Point2f> sourceLine;
    switch (lineType) {
        case LineType::SIDE_LINE_1:
            sourceLine = m_sourceLines.sideLine1;
            break;
        case LineType::END_LINE_2:
            sourceLine = m_sourceLines.endLine2;
            break;
        case LineType::SIDE_LINE_3:
            sourceLine = m_sourceLines.sideLine3;
            break;
        case LineType::THREE_POINT_LINE_4:
            sourceLine = m_sourceLines.threePointLine4;
            break;
        case LineType::LANE_LINE_5:
            sourceLine = m_sourceLines.laneLine5;
            break;
        case LineType::LANE_LINE_6:
            sourceLine = m_sourceLines.laneLine6;
            break;
        case LineType::FREE_THROW_LINE_7:
            sourceLine = m_sourceLines.freeThrowLine7;
            break;
        case LineType::FREE_THROW_CIRCLE_8:
            sourceLine = m_sourceLines.freeThrowCircle8;
            break;
        case LineType::FIELD_FILTER_LINES:
            sourceLine = m_sourceLines.fieldFilterLines;
            break;
        default:
            m_errorCode = ErrorCode::UNKNOWN_LINE_TYPE;
            return sourceLine;
    }
    std::vector<Point2f> mapped = mapLine(sourceLine, lineType);
    return mapped;
}

// 映射所有线条
std::vector<LineData> BasketballCourtMapper::mapAllLines() {
    std::vector<LineData> lines;

    // 定义所有线条的配置信息
    struct LineConfig {
        LineType type;
        std::string name;
        int tag;
    };

    std::vector<LineConfig> lineConfigs = {
        {LineType::SIDE_LINE_1, "sideLine", 1},
        {LineType::END_LINE_2, "endLine", 2},
        {LineType::SIDE_LINE_3, "sideLine", 3},
        {LineType::THREE_POINT_LINE_4, "threePointLine", 4},
        {LineType::LANE_LINE_5, "laneLine", 5},
        {LineType::LANE_LINE_6, "laneLine", 6},
        {LineType::FREE_THROW_LINE_7, "freeThrowLine", 7},
        {LineType::FREE_THROW_CIRCLE_8, "freeThrowCircle", 8},
        {LineType::FIELD_FILTER_LINES, "fieldFilterLines", 9} //场地过滤线，APP无需渲染
    };

    // 映射每条线
    for (const auto& config : lineConfigs) {
        LineData line = createBasketballLine(config.type, config.name, config.tag);

        // 检查线条映射是否成功（如果points为空，说明映射失败）
        // if (line.points.empty()) {
        //     // 如果映射失败，m_errorCode已经在mapLineByType中设置
        // }

        lines.push_back(line);
    }

    return lines;
}

// 创建单个线条
LineData BasketballCourtMapper::createBasketballLine(LineType lineType, const std::string& name, int tag) {
    LineData line;
    line.name = name;
    line.tag = tag;
    line.points = mapLineByType(lineType);
    return line;
}

// 初始化源关键点:A图
void BasketballCourtMapper::setSourceKeyPoints() {
    m_sourceKeyPoints = m_isHighschool ? m_sourceAllKeyPoints.highschoolPoints : m_sourceAllKeyPoints.standardPoints;
}

// 初始化源线条:A图
void BasketballCourtMapper::setSourceLines() {
    m_sourceLines = m_isHighschool ? initHighschoolLines() : initStandardLines();
}

// 获取A图预设的关键点数据: 模板A图程序写死，无需用户输入
void BasketballCourtMapper::initKeyPoints() {
    // 预设的A图关键点 Highschool 归一化坐标: 1732*1148
    m_sourceAllKeyPoints.highschoolPoints = {
        Point2f(0.116048, 0.078391),   // (201/1732, 90/1148)
        Point2f(0.365496, 0.076653),   // (633/1732, 88/1148)
        Point2f(0.596419, 0.072301),   // (1033/1732, 83/1148)
        Point2f(0.851617, 0.067944),   // (1475/1732, 78/1148)
        Point2f(0.369507, 0.580140),   // (640/1732, 666/1148)
        Point2f(0.597053, 0.581887)    // (1034/1732, 668/1148)
    };

    // 预设的A图关键点（standard），归一化：581*384
    m_sourceAllKeyPoints.standardPoints = {
        Point2f(0.070569, 0.007812),   // (41/581, 3/384)
        Point2f(0.337176, 0.002604),   // (196/581, 1/384)
        Point2f(0.659207, 0.007812),   // (383/581, 3/384)
        Point2f(0.934597, 0.013021),   // (543/581, 5/384)
        Point2f(0.344232, 0.580729),   // (200/581, 223/384)
        Point2f(0.659207, 0.586589)    // (383/581, 225/384)
    };

}

// 获取高中篮球场线条数据: Highschool
SourceBasketballLines BasketballCourtMapper::initHighschoolLines() {
    SourceBasketballLines sourceLines;

    sourceLines.sideLine1 = {
        Point2f(0.0318f, 0.0749f), Point2f(0.0306f, 0.2536f),
        Point2f(0.0294f, 0.4748f), Point2f(0.0318f, 0.6629f),
        Point2f(0.0318f, 0.8145f), Point2f(0.0323f, 0.9782f)
    };

    sourceLines.endLine2 = {
        Point2f(0.0323f, 0.0723f), Point2f(0.1161f, 0.0740f),
        Point2f(0.3695f, 0.0723f), Point2f(0.5954f, 0.0697f),
        Point2f(0.8477f, 0.0662f), Point2f(0.9382f, 0.0679f)
    };

    sourceLines.sideLine3 = {
        Point2f(0.9388f, 0.0653f), Point2f(0.9382f, 0.1681f),
        Point2f(0.9411f, 0.3032f), Point2f(0.9411f, 0.4362f),
        Point2f(0.9417f, 0.5998f), Point2f(0.9411f, 0.7937f)
    };

    sourceLines.threePointLine4 = {
        Point2f(0.1154f, 0.0723f), Point2f(0.1154f, 0.1838f),
        Point2f(0.1230f, 0.2875f), Point2f(0.1414f, 0.4060f),
        Point2f(0.1675f, 0.4905f), Point2f(0.1969f, 0.5557f),
        Point2f(0.2344f, 0.6078f), Point2f(0.2731f, 0.6560f),
        Point2f(0.3181f, 0.6950f), Point2f(0.3783f, 0.7344f),
        Point2f(0.4395f, 0.7504f), Point2f(0.5300f, 0.7545f),
        Point2f(0.6195f, 0.7212f), Point2f(0.6933f, 0.6560f),
        Point2f(0.7456f, 0.5933f), Point2f(0.7854f, 0.5183f),
        Point2f(0.8130f, 0.4487f), Point2f(0.8322f, 0.3858f),
        Point2f(0.8451f, 0.2831f), Point2f(0.8493f, 0.0618f)
    };

    sourceLines.laneLine5 = {
        Point2f(0.3672f, 0.0767f), Point2f(0.3701f, 0.1524f),
        Point2f(0.3695f, 0.2177f), Point2f(0.3683f, 0.2971f),
        Point2f(0.3660f, 0.3624f), Point2f(0.3683f, 0.4362f),
        Point2f(0.3683f, 0.5061f), Point2f(0.3695f, 0.5750f)
    };
    
    sourceLines.laneLine6 = {
        Point2f(0.5954f, 0.5775f),
        Point2f(0.5966f, 0.4991f), Point2f(0.5954f, 0.4425f),
        Point2f(0.5966f, 0.3799f), Point2f(0.5954f, 0.3057f),
        Point2f(0.5977f, 0.2361f), Point2f(0.5954f, 0.1742f),
        Point2f(0.5954f, 0.1228f), Point2f(0.5937f, 0.0679f)
    };

    sourceLines.freeThrowLine7 = {
        Point2f(0.3695f, 0.5750f),
        Point2f(0.4215f, 0.5801f), Point2f(0.4763f, 0.5819f),
        Point2f(0.5340f, 0.5845f), Point2f(0.5954f, 0.5775f)
    };

    sourceLines.freeThrowCircle8 = {
        Point2f(0.3701f, 0.5836f),
        Point2f(0.3753f, 0.6316f),
        Point2f(0.3875f, 0.6760f),
        Point2f(0.4104f, 0.7145f),
        Point2f(0.4395f, 0.7446f),
        Point2f(0.4808f, 0.7581f),
        Point2f(0.5209f, 0.7454f),
        Point2f(0.5543f, 0.7212f),
        Point2f(0.5751f, 0.6907f),
        Point2f(0.5892f, 0.6535f),
        Point2f(0.5966f, 0.6186f),
        Point2f(0.5966f, 0.5775f)
    };

    sourceLines.fieldFilterLines = {
        Point2f(0.0335f, 0.9307f),
        Point2f(0.0318f, 0.7110f),
        Point2f(0.0323f, 0.5227f),
        Point2f(0.0318f, 0.2509f),
        Point2f(0.0318f, 0.0706f),
        Point2f(0.1622f, 0.0723f),
        Point2f(0.2772f, 0.0740f),
        Point2f(0.4014f, 0.0723f),
        Point2f(0.5462f, 0.0697f),
        Point2f(0.6829f, 0.0706f),
        Point2f(0.8056f, 0.0653f),
        Point2f(0.9382f, 0.0618f),
        Point2f(0.9400f, 0.2308f),
        Point2f(0.9382f, 0.4756f),
        Point2f(0.9400f, 0.7090f),
        Point2f(0.9400f, 0.9350f)
    };

    return sourceLines;
}

// 获取标准篮球场线条数据: Standard
SourceBasketballLines BasketballCourtMapper::initStandardLines() {
    SourceBasketballLines sourceLines;

    // 边线1
    sourceLines.sideLine1 = {
        Point2f(0.0017f, 0.9362f), Point2f(0.0017f, 0.7536f), Point2f(0.0017f, 0.5826f),
        Point2f(0.0052f, 0.4435f), Point2f(0.0017f, 0.2986f), Point2f(0.0017f, 0.1449f),
        Point2f(0.0017f, 0.0029f)
    };

    // 底线2
    sourceLines.endLine2 = {
        Point2f(0.0017f, 0.0029f), Point2f(0.0706f, 0.0029f), Point2f(0.1652f, 0.0145f),
        Point2f(0.2685f, 0.0145f), Point2f(0.3528f, 0.0087f), Point2f(0.4561f, 0.0087f),
        Point2f(0.5680f, 0.0145f), Point2f(0.6936f, 0.0029f), Point2f(0.8193f, 0.0145f),
        Point2f(0.9122f, 0.0087f), Point2f(0.9862f, 0.0087f)
    };

    // 边线3
    sourceLines.sideLine3 = {
        Point2f(0.9914f, 0.0087f), Point2f(0.9914f, 0.2029f), Point2f(0.9948f, 0.3942f),
        Point2f(0.9948f, 0.5942f), Point2f(0.9914f, 0.7681f), Point2f(0.9914f, 0.9304f)
    };

    // 三分线4
    sourceLines.threePointLine4 = {
        Point2f(0.0654f, 0.0087f), Point2f(0.0620f, 0.1478f), Point2f(0.0620f, 0.3275f),
        Point2f(0.0861f, 0.4725f), Point2f(0.1256f, 0.5884f), Point2f(0.1738f, 0.6986f),
        Point2f(0.2324f, 0.7826f), Point2f(0.3012f, 0.8493f), Point2f(0.3666f, 0.8986f),
        Point2f(0.4475f, 0.9275f), Point2f(0.5077f, 0.9362f), Point2f(0.5697f, 0.9217f),
        Point2f(0.6368f, 0.8928f), Point2f(0.6971f, 0.8551f), Point2f(0.7453f, 0.8058f),
        Point2f(0.8003f, 0.7391f), Point2f(0.8279f, 0.6754f), Point2f(0.8692f, 0.6029f),
        Point2f(0.9002f, 0.5159f), Point2f(0.9260f, 0.4145f), Point2f(0.9380f, 0.3420f),
        Point2f(0.9346f, 0.1913f), Point2f(0.9346f, 0.0725f), Point2f(0.9312f, 0.0087f)
    };

    // 罚球占位线5
    sourceLines.laneLine5 = {
        Point2f(0.3408f, 0.0087f), Point2f(0.3408f, 0.1449f), Point2f(0.3408f, 0.2928f),
        Point2f(0.3373f, 0.4783f), Point2f(0.3373f, 0.6464f)
    };

    // 罚球占位线6
    sourceLines.laneLine6 = {
        Point2f(0.6627f, 0.6522f), Point2f(0.6592f, 0.5072f), Point2f(0.6592f, 0.3710f),
        Point2f(0.6627f, 0.2029f), Point2f(0.6592f, 0.0087f)
    };

    // 罚球线7
    sourceLines.freeThrowLine7 = {
        Point2f(0.3408f, 0.6522f), Point2f(0.3838f, 0.6522f), Point2f(0.4527f, 0.6551f),
        Point2f(0.5387f, 0.6522f), Point2f(0.6127f, 0.6551f), Point2f(0.6627f, 0.6551f)
    };

    // 罚球圈8
    sourceLines.freeThrowCircle8 = {
        Point2f(0.3838f, 0.6522f), Point2f(0.4010f, 0.7565f), Point2f(0.4355f, 0.8203f),
        Point2f(0.5009f, 0.8551f), Point2f(0.5645f, 0.8261f), Point2f(0.6041f, 0.7478f),
        Point2f(0.6162f, 0.6522f)
    };

    // 场地过滤线
    sourceLines.fieldFilterLines = {
        Point2f(0.0017f, 0.9014f), Point2f(0.0017f, 0.7188f),
        Point2f(0.0017f, 0.4638f), Point2f(0.0017f, 0.2058f),
        Point2f(0.0000f, 0.0029f), Point2f(0.1480f, 0.0029f),
        Point2f(0.3029f, 0.0087f), Point2f(0.4664f, 0.0087f),
        Point2f(0.6213f, 0.0029f), Point2f(0.7625f, 0.0087f),
        Point2f(0.8795f, 0.0087f), Point2f(0.9914f, 0.0087f),
        Point2f(0.9914f, 0.2174f), Point2f(0.9914f, 0.4493f),
        Point2f(0.9914f, 0.6667f), Point2f(0.9914f, 0.8986f)
    };

    return sourceLines;
}

// 坐标转换方法（归一化 <-> 像素）
Point2f BasketballCourtMapper::pixelToNormalized(const Point2i& pixelPt, int width, int height) {
    if (width <= 0 || height <= 0) {
        return Point2f(0.0f, 0.0f);
    }
    return Point2f(
        static_cast<float>(pixelPt.x) * 1.0 / width,
        static_cast<float>(pixelPt.y) * 1.0 / height
    );
}

Point2i BasketballCourtMapper::normalizedToPixel(const Point2f& normPt, int width, int height) {
    return Point2i(
        static_cast<int>(normPt.x * width),
        static_cast<int>(normPt.y * height)
    );
}

std::vector<Point2f> BasketballCourtMapper::pixelToNormalized(const std::vector<Point2i>& pixelPts, int width, int height) {
    std::vector<Point2f> normPts;
    for (const auto& pt : pixelPts) {
        normPts.push_back(pixelToNormalized(pt, width, height));
    }
    return normPts;
}

std::vector<Point2i> BasketballCourtMapper::normalizedToPixel(const std::vector<Point2f>& normPts, int width, int height) {
    std::vector<Point2i> pixelPts;
    for (const auto& pt : normPts) {
        pixelPts.push_back(normalizedToPixel(pt, width, height));
    }
    return pixelPts;
}

// 错误消息映射函数
ErrorInfo BasketballCourtMapper::getErrorInfo(ErrorCode code, Language lang) {
    ErrorInfo info;
    info.code = code;

    switch (code) {
        case ErrorCode::SUCCESS:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "成功" : "Success";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "操作成功完成" : "Operation completed successfully";
            break;

        case ErrorCode::INVALID_KEY_POINTS_COUNT:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点数量错误" : "Invalid key points count";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点数量必须为6个" : "Key points count must be 6";
            break;

        case ErrorCode::INSUFFICIENT_POINTS:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "点数不足" : "Insufficient points";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "打点点位数量不足2个" : "Insufficient marked points (less than 2)";
            break;

        case ErrorCode::INVALID_COORDINATE_RANGE:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "坐标范围错误" : "Invalid coordinate range";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点坐标必须在[0,1]范围内" : "Key point coordinates must be in [0,1] range";
            break;

        case ErrorCode::INVALID_KEY_POINTS_ORDER:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点顺序错误" : "Invalid key points order";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点顺序不符合要求" : "Key points order does not meet requirements";
            break;

        case ErrorCode::INVALID_KEY_POINTS_ORDER_1_2:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到1号点位和2号点位错误，请进行调整" : "Detected 1st and 2nd key points error, please adjust";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到1号点位和2号点位错误，请进行调整" : "Detected 1st and 2nd key points error, please adjust";
            break;

        case ErrorCode::INVALID_KEY_POINTS_ORDER_2_3:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到2号点位和3号点位错误，请进行调整" : "Detected 2nd and 3rd key points error, please adjust";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到2号点位和3号点位错误，请进行调整" : "Detected 2nd and 3rd key points error, please adjust";
            break;

        case ErrorCode::INVALID_KEY_POINTS_ORDER_3_4:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到3号点位和4号点位错误，请进行调整" : "Detected 3rd and 4th key points error, please adjust";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到3号点位和4号点位错误，请进行调整" : "Detected 3rd and 4th key points error, please adjust";
            break;

        case ErrorCode::INVALID_KEY_POINTS_ORDER_5_6:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到5号点位和6号点位错误，请进行调整" : "Detected 5th and 6th key points error, please adjust";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "检测到5号点位和6号点位错误，请进行调整" : "Detected 5th and 6th key points error, please adjust";
            break;

        case ErrorCode::KEY_POINTS_VALIDATION_FAILED:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点验证失败" : "Key points validation failed";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "关键点验证未通过" : "Key points validation did not pass";
            break;

        case ErrorCode::HOMOGRAPHY_CALCULATION_FAILED:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "单应性矩阵计算失败" : "Mapping calculation failed";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "计算单应性矩阵时出错" : "Error occurred while calculating mapping matrix";
            break;

        case ErrorCode::HOMOGRAPHY_MATRIX_INVALID:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "单应性矩阵无效" : "Invalid mapping matrix";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "单应性矩阵维度不正确" : "Mapping matrix dimension is incorrect";
            break;

        case ErrorCode::HOMOGRAPHY_MATRIX_SINGULAR:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "单应性矩阵奇异" : "Singular mapping matrix";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "单应性矩阵接近奇异，计算结果不可靠" : "Mapping matrix is near singular, calculation result unreliable";
            break;

        case ErrorCode::POINT_MAPPING_FAILED:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "点映射失败" : "Point mapping failed";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "点映射过程中出现错误" : "Error occurred during point mapping";
            break;

        case ErrorCode::INVALID_MAPPING_RESULT:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "映射结果无效" : "Invalid mapping result";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "点映射结果无效" : "Point mapping result is invalid";
            break;

        case ErrorCode::UNKNOWN_LINE_TYPE:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "未知线条类型" : "Unknown line type";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "指定的线条类型不存在" : "Specified line type does not exist";
            break;

        case ErrorCode::SVD_DECOMPOSITION_FAILED:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "SVD分解失败" : "SVD decomposition failed";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵SVD分解过程中出现错误" : "Error occurred during matrix SVD decomposition";
            break;

        case ErrorCode::MATRIX_DIMENSION_MISMATCH:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵维度不匹配" : "Matrix dimension mismatch";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵维度不一致，无法相乘" : "Matrix dimensions are inconsistent, cannot multiply";
            break;

        case ErrorCode::MATRIX_TOO_LARGE:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵过大" : "Matrix too large";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵过大，可能导致计算不稳定" : "Matrix is too large, may cause calculation instability";
            break;

        case ErrorCode::MATRIX_EMPTY:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "矩阵为空" : "Matrix is empty";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "输入矩阵为空" : "Input matrix is empty";
            break;

        case ErrorCode::UNKNOWN_ERROR:
        default:
            info.message = (lang == Language::CHINESE_SIMPLIFIED) ? "未知错误" : "Unknown error";
            info.description = (lang == Language::CHINESE_SIMPLIFIED) ? "发生了未知错误" : "An unknown error occurred";
            break;
    }

    return info;
}

std::string BasketballCourtMapper::getErrorMessage(ErrorCode code, Language lang) {
    return getErrorInfo(code, lang).message;
}

double BasketballCourtMapper::calculateDeterminant3x3(const std::vector<std::vector<double>>& matrix) {
    // 检查矩阵维度
    if (matrix.size() != 3) {
        m_errorCode = ErrorCode::MATRIX_DIMENSION_MISMATCH;
        return -1.0; // 使用-1.0明确表示错误
    }

    for (size_t i = 0; i < 3; ++i) {
        if (matrix[i].size() != 3) {
            m_errorCode = ErrorCode::MATRIX_DIMENSION_MISMATCH;
            return -1.0; // 使用-1.0明确表示错误
        }
    }

    // 使用拉普拉斯展开法计算行列式
    // det(A) = a00(a11a22 - a12a21) - a01(a10a22 - a12a20) + a02(a10a21 - a11a20)
    double det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
               - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
               + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

    // 检查计算结果是否有效
    if (std::isnan(det) || std::isinf(det)) {
        m_errorCode = ErrorCode::HOMOGRAPHY_CALCULATION_FAILED;
        return -1.0;
    }

    return det;
}


} // namespace XbotGoAiAnalysis
