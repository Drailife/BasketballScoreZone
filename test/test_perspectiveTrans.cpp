#include <iostream>
#include <vector>
#include "perspectiveTransform.h"
#include "ScoreStrategy.h"
using namespace XbotGoAiAnalysis;

int main() {
    // 原始像素点数据
// [[0.60572916 0.5750242 ]
//  [0.50104165 0.59535336]
//  [0.37291667 0.61858666]
//  [0.1890625  0.6573088 ]
//  [0.67291665 0.66118103]
//  [0.5161458  0.71926427]]
    const std::vector<Point2f> pts_A_pixel = {
        Point2f(0.6057291666666667f, 0.5750242013552759f, 1.0f),
        Point2f(0.5010416666666667f, 0.5953533397870281f, 1.0f),
        Point2f(0.3729166666666667f, 0.6185866408518877f, 1.0f),
        Point2f(0.1890625f, 0.6573088092933205f, 1.0f),
        Point2f(0.6729166666666666f, 0.6611810302734375f, 1.0f),
        Point2f(0.5161458333333334f, 0.7192642707824707f, 1.0f)
    };
// [[0.06884682 0.0115942 ]
//  [0.3425129  0.01449275]
//  [0.6643718  0.0115942 ]
//  [0.9345955  0.0057971 ]
//  [0.3425129  0.6463768 ]
//  [0.66092944 0.64927536]]
    const std::vector<Point2f> pts_B_pixel = {
        Point2f(0.06884681583476764f, 0.011594202898550725f, 1.0f),
        Point2f(0.342512908777969f, 0.014492753623188406f, 1.0f),
        Point2f(0.6643717728055077f, 0.011594202898550725f, 1.0f),
        Point2f(0.9345955249569707f, 0.005797101449275362f, 1.0f),
        Point2f(0.342512908777969f, 0.6463768115942029f, 1.0f),
        Point2f(0.6609294414520264f, 0.6492753623188406f, 1.0f)
    };


    try {
        const std::vector<std::vector<double>> H =
            PerspectiveTransform::findHomography(pts_A_pixel, pts_B_pixel);
        std::cout << "单应性矩阵 H:" << std::endl;
        for (const auto& row : H) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "计算单应性矩阵失败: " << e.what() << std::endl;
        return 1;
    }
    // 测试点映射
    const Point2f testPtA(0.5f, 0.5f);
    try {
        const std::vector<std::vector<double>> H =
            PerspectiveTransform::findHomography(pts_A_pixel, pts_B_pixel);
        const Point2f mappedPt = PerspectiveTransform::mapPointAToWarpedA(testPtA, H);
        std::cout << "映射后的点坐标: (" << mappedPt.x << ", " << mappedPt.y << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "点映射失败: " << e.what() << std::endl;
        return 1;
    }

    // 计分策略示例：初始化后，只需要输入人物脚点的相对坐标
    ScoreStrategy scoreStrategy(pts_A_pixel);
    if (!scoreStrategy.isReady()) {
        std::cerr << "计分策略初始化失败: " << scoreStrategy.lastError() << std::endl;
        return 1;
    }

    const Point2f playerFootPointNorm(0.50f, 0.78f);
    const ScoreResult scoreResult = scoreStrategy.judgePlayerPoint(playerFootPointNorm);
    std::cout << "人物脚点: (" << playerFootPointNorm.x << ", "
              << playerFootPointNorm.y << ")" << std::endl;
    std::cout << "模板映射点: (" << scoreResult.templatePoint.x << ", "
              << scoreResult.templatePoint.y << ")" << std::endl;
    std::string scoreLabel;
    if (scoreResult.score == 1) {
        scoreLabel = "1分（罚球线中点附近）";
    } else if (scoreResult.outsideThreeLine) {
        scoreLabel = "3分（三分线外）";
    } else {
        scoreLabel = "2分（三分线内）";
    }
    std::cout << "得分判定: " << scoreLabel << std::endl;
    std::cout << "分值: " << scoreResult.score << std::endl;
    return 0;
}
