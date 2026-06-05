#include "BasketballCourtMapper.h"
#include <iostream>
#include <iomanip>

using namespace XbotGoAiAnalysis;

void printPoints(const std::vector<Point2f>& points, const std::string& label, bool ifJSON = false) {
    std::cout << label << std::endl;
    std::cout << " {";
    if (ifJSON) {
        for (size_t i = 0; i < points.size(); ++i) {
            std::cout << "{ \"x\": " << points[i].x  << ", \"y\": " << points[i].y << "}, " ;
        }
    } else {
        for (size_t i = 0; i < points.size(); ++i) {
            std::cout << " {" << points[i].x << ", " << points[i].y << "}";
            if (i != points.size() - 1) {
                std::cout << ", ";
            }
        }
    }
    std::cout << "}" << std::endl;
    std::cout << std::endl;
}

// 打印新的JSON格式的篮球场地线条
void printBasketballLinesJSON(const HalfCourtLines& halfCourtLines, float imageWidth, float imageHeight) {
    std::cout << "{" << std::endl;
    std::cout << "    \"yaw\": " << halfCourtLines.angleYaw << "," << std::endl;
    std::cout << "    \"pitch\": " << halfCourtLines.anglePitch << "," << std::endl;
    std::cout << "    \"halfcourt\": \"" << halfCourtLines.halfCourtName << "\"," << std::endl;
    std::cout << "    \"lines\": [" << std::endl;
    
    for (size_t i = 0; i < halfCourtLines.lines.size(); ++i) {
        const auto& line = halfCourtLines.lines[i];
        std::cout << "        {" << std::endl;
        std::cout << "            \"name\": \"" << line.name << "\"," << std::endl;
        std::cout << "            \"tag\": " << line.tag << "," << std::endl;
        std::cout << "            \"points\": [" << std::endl;
        
        for (size_t j = 0; j < line.points.size(); ++j) {
            const auto& point = line.points[j];
            std::cout << "                {" << std::endl;
            std::cout << "                    \"x\": " << std::fixed << std::setprecision(6) << point.x << "," << std::endl;
            std::cout << "                    \"y\": " << std::fixed << std::setprecision(6) << point.y << std::endl;
            std::cout << "                }";
            if (j != line.points.size() - 1) {
                std::cout << ",";
            }
            std::cout << std::endl;
        }
        
        std::cout << "            ]" << std::endl;
        std::cout << "        }";
        if (i != halfCourtLines.lines.size() - 1) {
            std::cout << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "    ]" << std::endl;
    std::cout << "}" << std::endl;
}

// 单个半场打点映射模拟：
int main() {
    std::vector<Point2f> pointsBnorm; // 用户图像打的6个点，需用户打点
    // 用户图像打点输入模拟（需归一化到[0,1]）
    float imgBWidth = 1920.0; // 用户图像B宽度
    float imgBHeight = 1080.0; // 用户图像B高度

    // B图关键点(highschool)：需用户输入，仿真预设
    std::vector<Point2f> highschoolPointsB = {
        Point2f(851/imgBWidth, 428/imgBHeight), Point2f(1030/imgBWidth, 460/imgBHeight), Point2f(1303/imgBWidth, 495/imgBHeight),
        Point2f(1753/imgBWidth, 561/imgBHeight), Point2f(623/imgBWidth, 586/imgBHeight), Point2f(860/imgBWidth, 520/imgBHeight)
    };

    // std::vector<Point2f> highschoolPointsB = {
    //     Point2f(0.09842352f, 0.544697f),
    //     Point2f(0.3336174f, 0.5212121f),
    //     Point2f(0.56923735f, 0.5090909f),
    //     Point2f(0.8150831f, 0.46439394f),
    //     Point2f(0.5717938f, 0.79318184f),
    //     Point2f(0.707712f, 0.58257574f)
    // };

    // B图关键点(standard)：需用户输入，仿真预设
    std::vector<Point2f> standardPointsB = {
        Point2f(43/imgBWidth, 991/imgBHeight), Point2f(923/imgBWidth, 1006/imgBHeight), Point2f(1631/imgBWidth, 1015/imgBHeight),
        Point2f(2068/imgBWidth, 1026/imgBHeight), Point2f(1620/imgBWidth, 1203/imgBHeight), Point2f(2378/imgBWidth, 1178/imgBHeight)
    };

    // 输入参数配置
    MappingConfig config;
    config.isHighschoolMode = true;
    config.imageBWidth = imgBWidth;
    config.imageBHeight = imgBHeight;
    config.angleYaw = -50.0; // 用户当前画面角度示例
    config.anglePitch = 0.0;

    if (config.isHighschoolMode) {
        config.pointsBnorm = highschoolPointsB;
    } else {
        config.pointsBnorm = standardPointsB;
    }

    BasketballCourtMapper mapper; // 初始化映射：模板图场地线采样点

    // 映射，输出：MappingResult result
    // TODO：当前仅为半场打点，全场=2个半场？两个半场拼起来的效果？实时性？
    MappingResult result = mapper.basketballLineMappingRun(config);
    if (result.success) {
        std::cout << "篮球场地线映射成功！" << std::endl;

        // 打印新的JSON格式的篮球场地线条
        std::cout << "篮球场地线条映射结果（JSON格式）:" << std::endl;
        printBasketballLinesJSON(result.halfCourtLines, config.imageBWidth, config.imageBHeight);
    } else {
        std::cout << "篮球场地线映射失败：" << std::endl;
        std::cout << "错误代码: " << static_cast<int>(result.errorCode) << std::endl;
        std::cout << "错误消息: " << result.errorMessage << std::endl;
        
        // 获取详细的错误信息
        ErrorInfo errorInfo = BasketballCourtMapper::getErrorInfo(result.errorCode);
        std::cout << "错误描述: " << errorInfo.description << std::endl;
        
        // 演示多语言支持
        std::cout << "\n多语言错误消息示例:" << std::endl;
        std::cout << "中文简体: " << BasketballCourtMapper::getErrorMessage(result.errorCode, Language::CHINESE_SIMPLIFIED) << std::endl;
        std::cout << "English: " << BasketballCourtMapper::getErrorMessage(result.errorCode, Language::ENGLISH) << std::endl;
    }

    return 0;
}