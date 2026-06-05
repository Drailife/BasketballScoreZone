#ifndef PERSPECTIVETRANSFORM_H
#define PERSPECTIVETRANSFORM_H
#include <vector>

/*
实现功能：25.08.12 liaoxq
像素点结构体：Point2i, Point2f
公共接口：
1. 计算单应性矩阵：findHomography
2. 点映射函数：mapPointAToWarpedA
私有方法：
3. 矩阵乘法函数：matrixMultiply
4. 矩阵转置函数：matrixTranspose
5. SVD分解：svd
6. Jacobi SVD分解：JacobiSVD
*/

namespace XbotGoAiAnalysis{

// 坐标点结构体
struct Point2i {
    int x, y;
    Point2i() : x(0), y(0) {}
    Point2i(int x, int y) : x(x), y(y) {}
};

struct Point2f {
    float x, y, score;
    Point2f() : x(0.0f), y(0.0f), score(0.0f) {}
    Point2f(float x, float y) : x(x), y(y), score(0.0f) {}
    Point2f(float x, float y, float score) : x(x), y(y), score(score) {}
};

// 透视变换类
class PerspectiveTransform {
public:
    /**
     * @brief 计算单应性矩阵
     * @param srcPts 模板点集，至少4个点，目前设定6个点（程序内预设写好）
     * @param dstPts 用户图像点集，数量与srcPts一致，目前设定6个点（需用户手动打点）
     * @return 单应性矩阵（3x3 double型二维数组）
     */
    static std::vector<std::vector<double>> findHomography(const std::vector<Point2f>& srcPts, const std::vector<Point2f>& dstPts);
    
    /**
     * @brief 点映射函数，将模板点pt通过单应性矩阵M映射到用户目标图像
     * @param pt 模板点
     * @param M 单应性矩阵
     * @return 映射后的浮点坐标（用户图像上的点）
     */
    static Point2f mapPointAToWarpedA(const Point2f& pt, const std::vector<std::vector<double>>& M);

private:
    static std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& A, 
                                                           const std::vector<std::vector<double>>& B);
    
    static std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>>& A);
    
    static std::vector<std::vector<double>> svd(const std::vector<std::vector<double>>& matSrc);
    
    static void JacobiSVD(std::vector<std::vector<double>>& At, 
                         std::vector<std::vector<double>>& _W, 
                         std::vector<std::vector<double>>& Vt);
};

} //namespace

#endif // PERSPECTIVETRANSFORM_H
