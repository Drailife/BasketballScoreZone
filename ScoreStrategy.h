#ifndef SCORESTRATEGY_H
#define SCORESTRATEGY_H

#include "perspectiveTransform.h"
#include <string>
#include <vector>

namespace XbotGoAiAnalysis {

enum class ThreePointArea {
    UNKNOWN = 0,
    INSIDE = 1,
    OUTSIDE = 2
};

struct ScoreResult {
    bool success = false;
    int score = 2;
    bool outsideThreeLine = false;
    bool onThreeLine = false;
    ThreePointArea area = ThreePointArea::UNKNOWN;
    Point2f imagePoint;
    Point2f templatePoint;
    std::string region;
    std::string reason;
    double distanceToCenter = -1.0;
};

class ScoreStrategy {
public:
    ScoreStrategy() = default;
    explicit ScoreStrategy(const std::vector<Point2f>& imageKeyPoints);
    ScoreStrategy(const std::vector<Point2f>& imageKeyPoints,
                  const std::vector<bool>& validMask);

    static std::vector<Point2f> getStandardTemplateKeyPoints();

    bool setImageKeyPoints(const std::vector<Point2f>& imageKeyPoints);
    bool setHomography(const std::vector<std::vector<double>>& imageToTemplateH);

    bool isReady() const;
    std::string lastError() const;
    std::vector<std::vector<double>> getHomography() const;

    ScoreResult judgePlayerPoint(const Point2f& playerPointNorm,
                                 bool lineAsOutside = true) const;

    static ScoreResult judgePlayerPoint(
        const Point2f& playerPointNorm,
        const std::vector<Point2f>& imageKeyPoints,
        bool lineAsOutside = true);

    static ScoreResult judgeTemplatePoint(const Point2f& templatePointNorm,
                                          bool lineAsOutside = true);

    static Point2f footPointFromBox(const Point2f& leftTopNorm,
                                    const Point2f& rightBottomNorm);

private:
    bool buildHomography(const std::vector<Point2f>& imageKeyPoints);

    static bool isValidHomography(const std::vector<std::vector<double>>& H);
    static bool mapPointToTemplate(const Point2f& imagePointNorm,
                                   const std::vector<std::vector<double>>& H,
                                   Point2f& templatePointNorm);

    std::vector<std::vector<double>> m_imageToTemplateH;
    bool m_ready = false;
    std::string m_lastError;
};

} // namespace XbotGoAiAnalysis

#endif // SCORESTRATEGY_H
