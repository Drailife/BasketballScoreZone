import numpy as np

# 全局变量存储点坐标
points_A = []
points_B = []
current_points = None

def findHomography(src_pts, dst_pts):
    """使用最小二乘法计算单应性矩阵"""
    assert len(src_pts) == len(dst_pts) >= 4
    
    # 构建矩阵A
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    
    A = np.array(A)
    print(f"A: {A}")
    
    # 对A进行SVD分解
    U, S, Vt = np.linalg.svd(A)
    print(f"Vt: {Vt}")
    H = Vt[-1].reshape(3, 3)
    print(f"归一化前 H: {H}")
    
    # 归一化
    H = H / H[2, 2]
    print(f"归一化后 H: {H}")
    return H


# 定义一个映射函数，将img_A上的点映射到warped_A上的像素坐标
def map_point_A_to_warpedA(pt, M):
    """
    输入: pt - (x, y)格式的img_A上的点
    输出: (x', y')格式的warped_A上的点（浮点数）
    """
    pt_homo = np.array([[pt[0], pt[1], 1]], dtype=np.float32).T  # 3x1
    mapped = np.dot(M, pt_homo)
    mapped /= mapped[2, 0]
    return (mapped[0, 0], mapped[1, 0])


# 归一化函数
def norm_pt(pt, img):
    h, w = img.shape[:2]
    return (round(pt[0] / w, 4), round(pt[1] / h, 4))


def main():
    """
    模板打点记录：（highschool）
6个原始点：[(201, 90), (633, 88), (1033, 83), (1475, 78), (640, 666), (1034, 668)]

(standard):
6个原始点：[[(41, 3), (196, 1), (383, 3), (543, 5), (200, 223), (383, 225)]

    工程端说明：
    1. 设定的模板图的line坐标数组需归一化：因为工程端不同app平台使用的模板分辨率不同；
    2. 计算单应性矩阵时，需将归一化后的坐标转为像素坐标，再计算单应性矩阵（工程最优）；
    3. 对图像任一点进行映射时，需将归一化后的坐标转为像素坐标（单应性矩阵为像素坐标映射关系矩阵），再进行映射；
    """

    points_A =  [(201, 90), (633, 88), (1033, 83), (1475, 78), (640, 666), (1034, 668)]
    points_B =  [(851, 428), (1030, 460), (1303, 495), (1753, 561), (623, 520), (860, 586)]

    print(f"选择的A图点坐标（原始分辨率、归一化后）: {points_A}")
    print(f"选择的B图点坐标（原始分辨率、归一化后）: {points_B}")
    """
选择的A图点坐标（原始分辨率、归一化后）: [(0.1161, 0.0784), (0.3655, 0.0767), (0.5964, 0.0723), (0.8516, 0.0679), (0.3695, 0.5801), (0.597, 0.5819)]
选择的B图点坐标（原始分辨率、归一化后）: [(0.4432, 0.3963), (0.5365, 0.4259), (0.6786, 0.4583), (0.913, 0.5194), (0.3245, 0.4815), (0.4479, 0.5426)]
    """

    # 图像A的三分线：
    highschool_curve_threeline = [(200, 83), (200, 211), (213, 330), (245, 466), (290, 563), (341, 638), (406, 698), (473, 753), (551, 798), (655, 843), (761, 861), (918, 866), (1073, 828), (1201, 753), (1291, 681), (1360, 595), (1408, 515), (1441, 443), (1463, 325), (1471, 71)]

    print("highschool_curve_threeline:", highschool_curve_threeline)
    """
highschool_curve_threeline: [(0.1155, 0.0723), (0.1155, 0.1838), (0.123, 0.2875), (0.1415, 0.4059), (0.1674, 0.4904), (0.1969, 0.5557), (0.2344, 0.608), (0.2731, 0.6559), (0.3181, 0.6951), (0.3782, 0.7343), (0.4394, 0.75), (0.53, 0.7544), (0.6195, 0.7213), (0.6934, 0.6559), (0.7454, 0.5932), (0.7852, 0.5183), (0.8129, 0.4486), (0.832, 0.3859), (0.8447, 0.2831), (0.8493, 0.0618)]
    """


    # 【工程端计算】计算单应性矩阵（使用原始分辨率的坐标）
    # 计算像素坐标下的单应性矩阵
    M = findHomography(points_A, points_B)
    np.set_printoptions(precision=4, suppress=True) # 设置打印精度，log
    print("M:", M)
    

    # 实际操作：将img_A上的点映射到warped_A上
    print("warped 3line_points:")
    warped_line_points = []
    for pt in highschool_curve_threeline:
        # imgA的采样点为归一化坐标，需先转为imgA的像素坐标，再进行映射计算
        mapped_pt = map_point_A_to_warpedA(pt, M)
        warped_line_points.append(mapped_pt)
        print(f"{mapped_pt[0]}, {mapped_pt[1]}")



if __name__ == "__main__":
    main()