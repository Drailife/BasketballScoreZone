#include "perspectiveTransform.h"
#include "ScoreStrategy.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

using json = nlohmann::json;
using namespace XbotGoAiAnalysis;

int main() {
    // Read JSON file
    std::ifstream jsonFile("strategy/test_detections_score.json");
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

    // Extract keypoints (A-F)
    std::vector<Point2f> imageKeyPointsNorm;
    const auto& validKeypoints = data["valid_keypoints"];
    
    std::cout << "\n=== Keypoints (A-F) ===" << std::endl;
    for (const auto& kp : validKeypoints) {
        std::string label = kp["label"].get<std::string>();
        auto source = kp["source"];
        float x = source[0].get<float>() / imageWidth;
        float y = source[1].get<float>() / imageHeight;
        float conf = kp["conf"].get<float>();
        
        imageKeyPointsNorm.push_back(Point2f(x, y, conf));
        std::cout << label << ": (" << std::fixed << std::setprecision(4) 
                  << x << ", " << y << ") conf=" << conf << std::endl;
    }

    // Build homography
    ScoreStrategy scoreStrategy(imageKeyPointsNorm);
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
        
        auto footPoint = person["foot_point"];
        float footX = footPoint[0].get<float>() / imageWidth;
        float footY = footPoint[1].get<float>() / imageHeight;
        
        // Judge if person is outside three-line
        ScoreResult result = scoreStrategy.judgePlayerPoint(Point2f(footX, footY, conf), true);
        
        std::cout << "Person " << personId 
                  << " (conf=" << std::fixed << std::setprecision(3) << conf << ")"
                  << " | Foot: (" << footX << ", " << footY << ")" << std::endl;
        
        if (result.success) {
            std::cout << "  Region: " << result.region 
                      << " | Score: " << result.score
                      << " | Outside: " << (result.outsideThreeLine ? "YES (3pt)" : "NO (2pt)") << std::endl;
            
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
    int personOutside = 0;
    int personInside = 0;
    for (const auto& person : persons) {
        auto footPoint = person["foot_point"];
        float footX = footPoint[0].get<float>() / imageWidth;
        float footY = footPoint[1].get<float>() / imageHeight;
        float conf = person["conf"].get<float>();
        
        ScoreResult result = scoreStrategy.judgePlayerPoint(Point2f(footX, footY, conf), true);
        if (result.success) {
            if (result.outsideThreeLine) {
                personOutside++;
            } else {
                personInside++;
            }
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Persons outside 3-line: " << personOutside << std::endl;
    std::cout << "Persons inside 3-line: " << personInside << std::endl;
    
    return 0;
}