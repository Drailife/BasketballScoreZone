#include "perspectiveTransform.h"
#include "BasketballScoreJudge.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

using json = nlohmann::json;
using namespace XbotGoAiAnalysis;

int keypointIndexFromLabel(const std::string& label) {
    if (label == "A") return 0;
    if (label == "B") return 1;
    if (label == "C") return 2;
    if (label == "D") return 3;
    if (label == "E") return 4;
    if (label == "F") return 5;
    return -1;
}

Point2f normalizedPoint(float x, float y, float width, float height, float score = 0.0f) {
    return Point2f(x / width, y / height, score);
}

int main() {
    // Read JSON file
    std::ifstream jsonFile("python/testImgs/right_detections.json");
    if (!jsonFile.is_open()) {
        std::cerr << "Error: Cannot open JSON file" << std::endl;
        return 1;
    }

    json data;
    try {
        jsonFile >> data;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return 1;
    }

    // Extract image dimensions
    float imageWidth = data["image_width"].get<float>();
    float imageHeight = data["image_height"].get<float>();
    std::cout << "Image size: " << imageWidth << " x " << imageHeight << std::endl;

    // 只读取原始 keypoints。低置信点、缺失点由 BasketballScoreJudge 内部决定是否参与 H。
    std::vector<Point2f> imageKeyPointsNorm(6);
    const auto& keypoints = data["keypoints"];

    std::cout << "\n=== Raw Keypoints (A-F) ===" << std::endl;
    for (const auto& kp : keypoints) {
        std::string label = kp["label"].get<std::string>();
        int index = keypointIndexFromLabel(label);
        if (index < 0) {
            continue;
        }

        float x = kp["x"].get<float>();
        float y = kp["y"].get<float>();
        float conf = kp["conf"].get<float>();

        imageKeyPointsNorm[index] = normalizedPoint(x, y, imageWidth, imageHeight, conf);
        std::cout << label << ": (" << std::fixed << std::setprecision(4) 
                  << imageKeyPointsNorm[index].x << ", "
                  << imageKeyPointsNorm[index].y << ") conf=" << conf
                  << (conf < 0.5f ? "  -> filtered by strategy" : "")
                  << std::endl;
    }

    // Build homography
    BasketballScoreJudge scoreStrategy(imageKeyPointsNorm);
    if (!scoreStrategy.isReady()) {
        std::cerr << "Error: Failed to build homography - " << scoreStrategy.lastError() << std::endl;
        return 1;
    }
    std::cout << "\nHomography built successfully!" << std::endl;

    // Process persons
    const auto& persons = data["persons"];
    std::cout << "\n=== Person Three-Line Analysis ===" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    int personCount = 0;
    for (const auto& person : persons) {
        personCount++;
        int personId = person["id"].get<int>();
        float conf = person["conf"].get<float>();
        
        auto xyxy = person["xyxy"];
        Point2f leftTop = normalizedPoint(
            xyxy[0].get<float>(),
            xyxy[1].get<float>(),
            imageWidth,
            imageHeight);
        Point2f rightBottom = normalizedPoint(
            xyxy[2].get<float>(),
            xyxy[3].get<float>(),
            imageWidth,
            imageHeight);
        Point2f footPoint = BasketballScoreJudge::footPointFromBox(leftTop, rightBottom);

        // 外部只传人物框；脚点计算和内外线判断由 BasketballScoreJudge 接口完成。
        ScoreResult result = scoreStrategy.judgePlayerBox(leftTop, rightBottom, true);
        
        std::cout << "Person " << personId 
                  << " (conf=" << std::fixed << std::setprecision(3) << conf << ")"
                  << " | Foot: (" << footPoint.x << ", " << footPoint.y << ")"
                  << std::endl;
        
        if (result.success) {
            std::string scoreTag;
            if (result.score == 1) {
                scoreTag = "1PT";
            } else if (result.outsideThreeLine) {
                scoreTag = "3PT-OUT";
            } else {
                scoreTag = "2PT-IN";
            }

            std::cout << "  Region: " << result.region 
                      << " | Score: " << result.score
                      << " | Result: " << scoreTag << std::endl;
            
            if (result.distanceToCenter >= 0) {
                std::cout << "  Distance to center: " << std::fixed << std::setprecision(2) 
                          << result.distanceToCenter << std::endl;
            }
        } else {
            std::cout << "  Error: " << result.reason << std::endl;
        }
        std::cout << std::string(80, '-') << std::endl;
    }

    std::cout << "Total persons processed: " << personCount << std::endl;
    
    // Summary statistics
    int personOne = 0;
    int personOutside = 0;
    int personInside = 0;
    for (const auto& person : persons) {
        auto xyxy = person["xyxy"];
        Point2f leftTop = normalizedPoint(
            xyxy[0].get<float>(),
            xyxy[1].get<float>(),
            imageWidth,
            imageHeight);
        Point2f rightBottom = normalizedPoint(
            xyxy[2].get<float>(),
            xyxy[3].get<float>(),
            imageWidth,
            imageHeight);

        ScoreResult result = scoreStrategy.judgePlayerBox(leftTop, rightBottom, true);
        if (result.success) {
            if (result.score == 1) {
                personOne++;
            } else if (result.outsideThreeLine) {
                personOutside++;
            } else {
                personInside++;
            }
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Persons in 1-point area: " << personOne << std::endl;
    std::cout << "Persons outside 3-line: " << personOutside << std::endl;
    std::cout << "Persons inside 3-line: " << personInside << std::endl;
    
    return 0;
}
