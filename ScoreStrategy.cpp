#include "ScoreStrategy.h"
#include <cmath>

namespace XbotGoAiAnalysis {
namespace {

constexpr int kTemplatePointCount = 6; // 模板点数量
constexpr float kMinKeyPointScore = 0.5f;
constexpr double kTemplateWidth = 581.0; // 模板图宽度（单位：像素）
constexpr double kTemplateHeight = 345.0; // 模板图高度（单位：像素）

constexpr double kThreeLineCenterX = 290.5; // 三分线圆心X坐标（单位：像素）
constexpr double kThreeLineCenterY = 67.275; // 三分线圆心Y坐标（单位：像素）
constexpr double kThreeLineLeftX = 37.765; // 三分线左侧X坐标（单位：像素）
constexpr double kThreeLineRightX = 543.235; // 三分线右侧X坐标（单位：像素）
constexpr double kThreeLineCornerY = 117.3; // 三分线角落Y坐标（单位：像素）
constexpr double kThreeLineRadius = 255.0; // 三分线半径（单位：像素）
constexpr double kThreeLineEps = 10.0; // 三分线判断误差（单位：像素）
constexpr double kFreeThrowCenterX = 290.7905; // 罚球线中点X坐标（单位：像素）
constexpr double kFreeThrowCenterY = 224.250; // 罚球线中点Y坐标（单位：像素）
constexpr double kFreeThrowOnePointRadius = 25; // 判定1分的距离阈值（单位：像素）

bool isFinite(double value) {
    return std::isfinite(value);
}

bool isValidNormalizedPoint(const Point2f& point) {
    return isFinite(point.x) && isFinite(point.y) &&
           point.x >= 0.0f && point.x <= 1.0f &&
           point.y >= 0.0f && point.y <= 1.0f;
}

bool isMissingPoint(const Point2f& point) {
    return point.x == 0.0f && point.y == 0.0f;
}

bool shouldSkipByScore(const Point2f& point) {
    return point.score < kMinKeyPointScore;
}

ScoreResult defaultTwoPointResult(const Point2f& imagePoint,
                                  const std::string& reason) {
    ScoreResult result;
    result.success = false;
    result.score = 2;
    result.nearFreeThrowCenter = false;
    result.outsideThreeLine = false;
    result.area = ThreePointArea::UNKNOWN;
    result.imagePoint = imagePoint;
    result.reason = reason;
    return result;
}

} // namespace

ScoreStrategy::ScoreStrategy(const std::vector<Point2f>& imageKeyPoints) {
    setImageKeyPoints(imageKeyPoints);
}

std::vector<Point2f> ScoreStrategy::getStandardTemplateKeyPoints() {
    return {
        Point2f(0.06884681583476764f, 0.011594202898550725f, 1.0f),
        Point2f(0.342512908777969f, 0.014492753623188406f, 1.0f),
        Point2f(0.6643717728055077f, 0.011594202898550725f, 1.0f),
        Point2f(0.9345955249569707f, 0.005797101449275362f, 1.0f),
        Point2f(0.342512908777969f, 0.6463768115942029f, 1.0f),
        Point2f(0.6609294414520264f, 0.6492753623188406f, 1.0f)
    };
}

bool ScoreStrategy::setImageKeyPoints(const std::vector<Point2f>& imageKeyPoints) {
    return buildHomography(imageKeyPoints);
}

bool ScoreStrategy::setHomography(const std::vector<std::vector<double>>& imageToTemplateH) {
    if (!isValidHomography(imageToTemplateH)) {
        m_imageToTemplateH.clear();
        m_ready = false;
        m_lastError = "INVALID_HOMOGRAPHY";
        return false;
    }

    m_imageToTemplateH = imageToTemplateH;
    m_ready = true;
    m_lastError.clear();
    return true;
}

bool ScoreStrategy::isReady() const {
    return m_ready;
}

const std::string& ScoreStrategy::lastError() const {
    return m_lastError;
}

const std::vector<std::vector<double>>& ScoreStrategy::getHomography() const {
    return m_imageToTemplateH;
}

ScoreResult ScoreStrategy::judgePlayerPoint(const Point2f& playerPointNorm,
                                            bool lineAsOutside) const {
    if (!isValidNormalizedPoint(playerPointNorm)) {
        return defaultTwoPointResult(playerPointNorm, "INVALID_PLAYER_POINT");
    }

    if (!m_ready) {
        return defaultTwoPointResult(playerPointNorm,
                                     "HOMOGRAPHY_NOT_READY_DEFAULT_TWO_POINTS");
    }

    Point2f templatePoint;
    if (!mapPointToTemplate(playerPointNorm, m_imageToTemplateH, templatePoint)) {
        return defaultTwoPointResult(playerPointNorm,
                                     "POINT_MAPPING_FAILED_DEFAULT_TWO_POINTS");
    }

    ScoreResult result = judgeTemplatePoint(templatePoint, lineAsOutside);
    result.imagePoint = playerPointNorm;
    return result;
}

ScoreResult ScoreStrategy::judgePlayerBox(const Point2f& leftTopNorm,
                                          const Point2f& rightBottomNorm,
                                          bool lineAsOutside) const {
    return judgePlayerPoint(
        footPointFromBox(leftTopNorm, rightBottomNorm),
        lineAsOutside);
}

ScoreResult ScoreStrategy::judgePlayerPoint(
    const Point2f& playerPointNorm,
    const std::vector<Point2f>& imageKeyPoints,
    bool lineAsOutside) {
    ScoreStrategy strategy(imageKeyPoints);
    return strategy.judgePlayerPoint(playerPointNorm, lineAsOutside);
}

ScoreResult ScoreStrategy::judgeTemplatePoint(const Point2f& templatePointNorm,
                                              bool lineAsOutside) {
    ScoreResult result;
    result.success = true;
    result.templatePoint = templatePointNorm;
    result.reason = "OK";

    const double x = static_cast<double>(templatePointNorm.x) * kTemplateWidth;
    const double y = static_cast<double>(templatePointNorm.y) * kTemplateHeight;

    bool outside = false;
    bool onLine = false;

    result.distanceToFreeThrowCenter = std::hypot(
        x - kFreeThrowCenterX,
        y - kFreeThrowCenterY);
    if (result.distanceToFreeThrowCenter <= kFreeThrowOnePointRadius) {
        result.region = "free_throw_center_area";
        result.nearFreeThrowCenter = true;
        result.outsideThreeLine = false;
        result.onThreeLine = false;
        result.area = ThreePointArea::INSIDE;
        result.score = 1;
        return result;
    }

    if (y <= kThreeLineCornerY) {
        result.region = "corner_line_area";
        const bool onLeftLine = std::abs(x - kThreeLineLeftX) <= kThreeLineEps;
        const bool onRightLine = std::abs(x - kThreeLineRightX) <= kThreeLineEps;
        onLine = onLeftLine || onRightLine;

        if (lineAsOutside) {
            outside = x <= kThreeLineLeftX + kThreeLineEps ||
                      x >= kThreeLineRightX - kThreeLineEps;
        } else {
            outside = x < kThreeLineLeftX - kThreeLineEps ||
                      x > kThreeLineRightX + kThreeLineEps;
        }
    } else {
        result.region = "arc_area";
        result.distanceToCenter = std::hypot(
            x - kThreeLineCenterX,
            y - kThreeLineCenterY);
        onLine = std::abs(result.distanceToCenter - kThreeLineRadius) <=
                 kThreeLineEps;

        if (lineAsOutside) {
            outside = result.distanceToCenter >= kThreeLineRadius - kThreeLineEps;
        } else {
            outside = result.distanceToCenter > kThreeLineRadius + kThreeLineEps;
        }
    }

    result.onThreeLine = onLine;
    result.nearFreeThrowCenter = false;
    result.outsideThreeLine = outside;
    result.area = outside ? ThreePointArea::OUTSIDE : ThreePointArea::INSIDE;
    result.score = outside ? 3 : 2;
    return result;
}

Point2f ScoreStrategy::footPointFromBox(const Point2f& leftTopNorm,
                                        const Point2f& rightBottomNorm) {
    return Point2f(
        (leftTopNorm.x + rightBottomNorm.x) * 0.5f,
        rightBottomNorm.y);
}

bool ScoreStrategy::buildHomography(const std::vector<Point2f>& imageKeyPoints) {
    m_imageToTemplateH.clear();
    m_ready = false;
    m_lastError.clear();

    if (imageKeyPoints.size() != kTemplatePointCount) {
        m_lastError = "INVALID_KEY_POINTS_COUNT";
        return false;
    }

    const std::vector<Point2f> templateKeyPoints = getStandardTemplateKeyPoints();
    std::vector<Point2f> srcPoints;
    std::vector<Point2f> dstPoints;
    srcPoints.reserve(kTemplatePointCount);
    dstPoints.reserve(kTemplatePointCount);

    for (std::size_t i = 0; i < imageKeyPoints.size(); ++i) {
        if (shouldSkipByScore(imageKeyPoints[i])) {
            continue;
        }

        if (!isValidNormalizedPoint(imageKeyPoints[i])) {
            m_lastError = "INVALID_KEY_POINT_RANGE";
            return false;
        }

        if (isMissingPoint(imageKeyPoints[i])) {
            continue;
        }

        srcPoints.push_back(imageKeyPoints[i]);
        dstPoints.push_back(templateKeyPoints[i]);
    }

    if (srcPoints.size() < 4) {
        m_lastError = "INSUFFICIENT_KEY_POINTS_DEFAULT_TWO_POINTS";
        return false;
    }

    try {
        return setHomography(PerspectiveTransform::findHomography(srcPoints, dstPoints));
    } catch (const std::exception& error) {
        m_lastError = error.what();
        m_ready = false;
        m_imageToTemplateH.clear();
        return false;
    }
}

bool ScoreStrategy::isValidHomography(const std::vector<std::vector<double>>& H) {
    if (H.size() != 3) {
        return false;
    }

    for (const std::vector<double>& row : H) {
        if (row.size() != 3) {
            return false;
        }

        for (double value : row) {
            if (!isFinite(value)) {
                return false;
            }
        }
    }

    return true;
}

bool ScoreStrategy::mapPointToTemplate(const Point2f& imagePointNorm,
                                       const std::vector<std::vector<double>>& H,
                                       Point2f& templatePointNorm) {
    if (!isValidHomography(H)) {
        return false;
    }

    const double x = imagePointNorm.x;
    const double y = imagePointNorm.y;
    const double w = H[2][0] * x + H[2][1] * y + H[2][2];
    if (std::abs(w) < 1e-10) {
        return false;
    }

    const double mappedX = (H[0][0] * x + H[0][1] * y + H[0][2]) / w;
    const double mappedY = (H[1][0] * x + H[1][1] * y + H[1][2]) / w;
    if (!isFinite(mappedX) || !isFinite(mappedY)) {
        return false;
    }

    templatePointNorm = Point2f(
        static_cast<float>(mappedX),
        static_cast<float>(mappedY));
    return true;
}

} // namespace XbotGoAiAnalysis
