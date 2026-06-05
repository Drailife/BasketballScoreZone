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

// 单个人物的计分结果。
struct ScoreResult {
    bool success = false;                  // true 表示完成了有效映射和判断
    int score = 2;                         // 失败或点不足时默认 2 分
    bool nearFreeThrowCenter = false;      // 是否在罚球点附近的一分区域
    bool outsideThreeLine = false;         // 是否在三分线外
    bool onThreeLine = false;              // 是否贴近三分线
    ThreePointArea area = ThreePointArea::UNKNOWN;
    Point2f imagePoint;                    // 输入的人物脚点，原图归一化坐标
    Point2f templatePoint;                 // 映射后的模板归一化坐标
    std::string region;                    // free_throw_center_area / arc_area / corner_line_area
    std::string reason;                    // OK 或失败原因
    double distanceToCenter = -1.0;         // 到三分线圆心距离，模板像素坐标
    double distanceToFreeThrowCenter = -1.0;// 到罚球点距离，模板像素坐标
};

// 计分策略：先用 A-F 关键点建立“原图 -> 模板图”的单应性矩阵，
// 后续只需要传人物脚点归一化坐标，即可返回 1/2/3 分。
class ScoreStrategy {
public:
    ScoreStrategy() = default;
    explicit ScoreStrategy(const std::vector<Point2f>& imageKeyPoints);

    static std::vector<Point2f> getStandardTemplateKeyPoints();

    // imageKeyPoints 必须按 A-F 顺序传入。Point2f.score:
    //   >0 且 <0.5 会被过滤；
    //   ==0 视为未提供置信度，仅按坐标参与计算。
    bool setImageKeyPoints(const std::vector<Point2f>& imageKeyPoints);
    bool setHomography(const std::vector<std::vector<double>>& imageToTemplateH);

    bool isReady() const;
    const std::string& lastError() const;
    const std::vector<std::vector<double>>& getHomography() const;

    // 最常用入口：传人物脚点的原图归一化坐标。
    ScoreResult judgePlayerPoint(const Point2f& playerPointNorm,
                                 bool lineAsOutside = true) const;

    // 便捷入口：传人物框左上和右下归一化坐标，内部自动取底边中心脚点。
    ScoreResult judgePlayerBox(const Point2f& leftTopNorm,
                               const Point2f& rightBottomNorm,
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
