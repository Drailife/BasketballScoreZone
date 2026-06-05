#include "perspectiveTransform.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <limits>

namespace XbotGoAiAnalysis{

/*
核心算法组件: 单应性矩阵和透视变换方法，实现基于采样>=4个点，进行全场场地线映射。
*/
// 点映射函数
Point2f PerspectiveTransform::mapPointAToWarpedA(const Point2f& pt, const std::vector<std::vector<double>>& M) {
    // 构建齐次坐标 [x, y, 1]^T
    std::vector<std::vector<double>> ptHomo = {
        {static_cast<double>(pt.x)}, 
        {static_cast<double>(pt.y)}, 
        {1.0}
    };
    
    // 计算 M * ptHomo
    std::vector<std::vector<double>> mapped = matrixMultiply(M, ptHomo);

    // 归一化
    double w = mapped[2][0];
    if (std::abs(w) > 1e-10) {
        double x = static_cast<double>(mapped[0][0] * 1.0 / w);
        double y = static_cast<double>(mapped[1][0] * 1.0 / w);
        return Point2f(x, y);
    } else {
        return Point2f(0.0f, 0.0f);
    }
}

// 矩阵乘法函数
std::vector<std::vector<double>> PerspectiveTransform::matrixMultiply(const std::vector<std::vector<double>>& A, 
                                                const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    
    if (colsA != rowsB) {
        throw std::runtime_error("MATRIX_DIMENSION_MISMATCH");
    }
    
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}

// 矩阵转置函数
std::vector<std::vector<double>> PerspectiveTransform::matrixTranspose(const std::vector<std::vector<double>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0.0));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }
    
    return result;
}

// 计算单应性矩阵：srcPts=A图像6个点，dstPts=B图像6个点；返回H矩阵3x3
std::vector<std::vector<double>> PerspectiveTransform::findHomography(const std::vector<Point2f>& srcPts,
                                               const std::vector<Point2f>& dstPts) {
    // 输入验证
    if (srcPts.empty() || dstPts.empty()) {
        throw std::runtime_error("MATRIX_EMPTY");
    }
    
    if (srcPts.size() != dstPts.size()) {
        throw std::runtime_error("INVALID_KEY_POINTS_COUNT");
    }
    
    if (srcPts.size() < 4) {
        throw std::runtime_error("INVALID_KEY_POINTS_COUNT");
    }
    
    // 验证归一化坐标范围
    for (const auto& pt : srcPts) {
        if (pt.x < 0.0f || pt.x > 1.0f || pt.y < 0.0f || pt.y > 1.0f) {
            throw std::runtime_error("INVALID_COORDINATE_RANGE");
        }
    }
    
    for (const auto& pt : dstPts) {
        if (pt.x < 0.0f || pt.x > 1.0f || pt.y < 0.0f || pt.y > 1.0f) {
            throw std::runtime_error("INVALID_COORDINATE_RANGE");
        }
    }
    
    // 构建矩阵A
    std::vector<std::vector<double>> A;
    for (size_t i = 0; i < srcPts.size(); i++) {
        double x = srcPts[i].x;
        double y = srcPts[i].y;
        double u = dstPts[i].x;
        double v = dstPts[i].y;
        
        std::vector<double> row1 = {x, y, 1, 0, 0, 0, -u*x, -u*y, -u};
        std::vector<double> row2 = {0, 0, 0, x, y, 1, -v*x, -v*y, -v};
        
        A.push_back(row1);
        A.push_back(row2);
    }
    
    // 对A进行SVD分解
    std::vector<std::vector<double>> Vt = svd(A);
    if (Vt.empty()) {
        throw std::runtime_error("SVD_DECOMPOSITION_FAILED");
    }
    
    // 取Vt的最后一行作为H矩阵
    std::vector<std::vector<double>> H(3, std::vector<double>(3, 0.0));
    int lastRow = Vt.size() - 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            H[i][j] = Vt[lastRow][i * 3 + j];
        }
    }

    // 归一化 - 增加数值稳定性检查
    double h22 = H[2][2];
    const double EPSILON = 1e-10;
    if (std::abs(h22) > EPSILON) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                H[i][j] /= h22;
            }
        }
    } else {
        throw std::runtime_error("HOMOGRAPHY_MATRIX_SINGULAR");
    }

    return H;
}

// matSrc为原始矩阵，支持非方阵，返回matVt
std::vector<std::vector<double>> PerspectiveTransform::svd(const std::vector<std::vector<double>>& matSrc)
{
    // 输入验证
    if (matSrc.empty()) {
        throw std::runtime_error("MATRIX_EMPTY");
    }
    
    if (matSrc[0].empty()) {
        throw std::runtime_error("MATRIX_EMPTY");
    }
    
    int m = matSrc.size();
    int n = matSrc[0].size();
    
    // 检查矩阵维度一致性
    for (const auto& row : matSrc) {
        if (row.size() != static_cast<size_t>(n)) {
            throw std::runtime_error("MATRIX_DIMENSION_MISMATCH");
        }
    }
    
    // 检查矩阵大小限制
    if (m > 1000 || n > 1000) {
        throw std::runtime_error("MATRIX_TOO_LARGE");
    }

    // matD存放奇异值，matU存放左奇异向量，matVt存放转置的右奇异向量
    std::vector<std::vector<double>> matD, matU, matVt;

    bool at = false;
    int mm = m, nn = n;
    if (m < n) {
        std::swap(mm, nn);
        at = true;
    }

    matD.assign(nn, std::vector<double>(1, 0.0));
    matU.assign(mm, std::vector<double>(mm, 0.0));
    matVt.assign(nn, std::vector<double>(nn, 0.0));
    std::vector<std::vector<double>> tmpU = matU, tmpV = matVt;

    std::vector<std::vector<double>> tmpA, tmpA_;
    if (!at)
        tmpA = matrixTranspose(matSrc);
    else
        tmpA = matSrc;

    if (mm == nn) {
        tmpA_ = tmpA;
    }
    else {
        tmpA_.assign(mm, std::vector<double>(mm, 0.0));
        for (int i = 0; i < nn; ++i) {
            std::copy(tmpA[i].begin(), tmpA[i].end(), tmpA_[i].begin());
        }
    }
    JacobiSVD(tmpA_, matD, tmpV);

    if (!at) {
        matU = matrixTranspose(tmpA_);
        matVt = tmpV;
    }
    else {
        matU = matrixTranspose(tmpV);
        matVt = tmpA_;
    }

    // 验证SVD结果的有效性
    if (matVt.empty() || matVt[0].empty()) {
        throw std::runtime_error("SVD_DECOMPOSITION_FAILED");
    }

    return matVt;
}

// svd 分解函数：
void PerspectiveTransform::JacobiSVD(std::vector<std::vector<double>>& At,
    std::vector<std::vector<double>>& _W, std::vector<std::vector<double>>& Vt)
{
    double minval = std::numeric_limits<double>::min();
    double eps = std::numeric_limits<double>::epsilon() * 2;
    const int m = At[0].size();
    const int n = _W.size();
    const int n1 = m; // urows
    std::vector<double> W(n, 0.);

    // 初始化W和Vt
    for (int i = 0; i < n; i++) {
        double sd = 0.;
        for (int k = 0; k < m; k++) {
            double t = At[i][k];
            sd += t * t;
        }
        W[i] = sd;
        std::fill(Vt[i].begin(), Vt[i].end(), 0);
        Vt[i][i] = 1;
    }

    int maxIter = std::max(m, 30);
    for (int iter = 0; iter < maxIter; iter++) {
        bool changed = false;
        double c, s;

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                double* Ai = At[i].data();
                double* Aj = At[j].data();
                double a = W[i], p = 0, b = W[j];

                for (int k = 0; k < m; k++)
                    p += Ai[k] * Aj[k];

                if (std::abs(p) <= eps * std::sqrt(a * b))
                    continue;

                p *= 2;
                double beta = a - b;
                double gamma = std::hypot(p, beta);
                if (gamma == 0) continue; // 防止除零

                if (beta < 0) {
                    double delta = (gamma - beta) * 0.5;
                    s = std::sqrt(delta / gamma);
                    c = p / (gamma * s * 2);
                }
                else {
                    c = std::sqrt((gamma + beta) / (gamma * 2));
                    s = p / (gamma * c * 2);
                }

                a = b = 0;
                for (int k = 0; k < m; k++) {
                    double t0 = c * Ai[k] + s * Aj[k];
                    double t1 = -s * Ai[k] + c * Aj[k];
                    Ai[k] = t0; Aj[k] = t1;
                    a += t0 * t0; b += t1 * t1;
                }
                W[i] = a; W[j] = b;

                changed = true;

                double* Vi = Vt[i].data();
                double* Vj = Vt[j].data();

                for (int k = 0; k < n; k++) {
                    double t0 = c * Vi[k] + s * Vj[k];
                    double t1 = -s * Vi[k] + c * Vj[k];
                    Vi[k] = t0; Vj[k] = t1;
                }
            }
        }

        if (!changed)
            break;
    }

    // 计算奇异值
    for (int i = 0; i < n; i++) {
        double sd = 0.;
        for (int k = 0; k < m; k++) {
            double t = At[i][k];
            sd += t * t;
        }
        W[i] = std::sqrt(sd);
    }

    // 排序
    for (int i = 0; i < n - 1; i++) {
        int j = i;
        for (int k = i + 1; k < n; k++) {
            if (W[j] < W[k])
                j = k;
        }
        if (i != j) {
            std::swap(W[i], W[j]);
            for (int k = 0; k < m; k++)
                std::swap(At[i][k], At[j][k]);
            for (int k = 0; k < n; k++)
                std::swap(Vt[i][k], Vt[j][k]);
        }
    }

    for (int i = 0; i < n; i++)
        _W[i][0] = W[i];

    std::srand((unsigned int)std::time(nullptr));

    // 归一化左奇异向量
    for (int i = 0; i < n1; i++) {
        double sd = i < n ? W[i] : 0;

        for (int ii = 0; ii < 100 && sd <= minval; ii++) {
            // 若奇异值为0，生成随机向量并正交化
            const double val0 = 1.0 / m;
            for (int k = 0; k < m; k++) {
                unsigned int rng = std::rand();
                double val = (rng & 256) != 0 ? val0 : -val0;
                At[i][k] = val;
            }
            for (int iter = 0; iter < 2; iter++) {
                for (int j = 0; j < i; j++) {
                    double dot = 0;
                    for (int k = 0; k < m; k++)
                        dot += At[i][k] * At[j][k];
                    double asum = 0;
                    for (int k = 0; k < m; k++) {
                        double t = At[i][k] - dot * At[j][k];
                        At[i][k] = t;
                        asum += std::abs(t);
                    }
                    asum = asum > eps * 100 ? 1 / asum : 0;
                    for (int k = 0; k < m; k++)
                        At[i][k] *= asum;
                }
            }

            sd = 0;
            for (int k = 0; k < m; k++) {
                double t = At[i][k];
                sd += t * t;
            }
            sd = std::sqrt(sd);
        }

        double s = (sd > minval ? 1 / sd : 0.);
        for (int k = 0; k < m; k++)
            At[i][k] *= s;
    }
}

} //using namespace
