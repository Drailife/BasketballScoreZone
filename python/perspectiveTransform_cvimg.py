import cv2
import numpy as np
import os

# 全局变量存储点坐标
points_A = []
points_B = []
current_points = None
current_image = None
current_scale = 1.0  # 添加缩放比例变量
window_name = ""
NUM_POINTS = 4 # 固定需要选择的点数

def click_event1(event, x, y, flags, param):
    """鼠标点击事件处理函数"""
    global current_points, current_image, current_scale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < NUM_POINTS:
            # 将显示坐标转换为原始图像坐标
            original_x = int(x / current_scale)
            original_y = int(y / current_scale)
            
            # 记录原始坐标
            current_points.append((original_x, original_y))
            
            # 在显示图像上画圆标记（使用显示坐标）
            cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(current_image, str(len(current_points)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, current_image)

def click_event(event, x, y, flags, param):
    """鼠标点击事件处理函数"""
    global current_points, current_image, current_scale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < NUM_POINTS:
            # 将显示坐标转换为原始图像坐标
            original_x = int(x / current_scale)
            original_y = int(y / current_scale)
            
            # 记录原始坐标
            current_points.append((original_x, original_y))
            
            # 在显示图像上画圆标记（使用显示坐标）
            cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(current_image, str(len(current_points)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, current_image)

def select_points(image, title, scale=0.5):
    """在图像上选择点并返回"""
    global current_points, current_image, window_name, current_scale
    
    current_points = []
    current_scale = scale
    window_name = title
    
    # 缩放图像用于显示
    height, width = image.shape[:2]
    display_width = int(width * scale)
    display_height = int(height * scale)
    current_image = cv2.resize(image, (display_width, display_height))
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    cv2.imshow(window_name, current_image)
    while len(current_points) < NUM_POINTS:
        cv2.waitKey(0)
        # 检查窗口是否被关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 1:
        cv2.destroyWindow(window_name)
    return current_points

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
    
    # 对A进行SVD分解
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # 归一化
    H = H / H[2, 2]
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

def map_points_with_boundary_filter(points, M, img_B_shape):
    """
    映射点列表，保留所有画面内的有效点位和距离有效点位最接近的超出画面的点位
    输入: points - 点列表 [(x1,y1), (x2,y2), ...]
          M - 单应性矩阵
          img_B_shape - 目标图像尺寸 (height, width)
    输出: 过滤后的点列表
    """
    height, width = img_B_shape[:2]
    valid_points = []  # 画面内的有效点
    out_of_bounds_points = []  # 超出画面的点
    
    # 先映射所有点并分类
    for pt in points:
        mapped_pt = map_point_A_to_warpedA(pt, M)
        x, y = mapped_pt
        
        if 0 <= x < width and 0 <= y < height:
            # 在画面内的点
            valid_points.append(mapped_pt)
        else:
            # 超出画面的点，计算到画面边界的距离
            x_clamped = max(0, min(x, width - 1))
            y_clamped = max(0, min(y, height - 1))
            distance = np.sqrt((x - x_clamped)**2 + (y - y_clamped)**2)
            out_of_bounds_points.append((mapped_pt, distance, (x_clamped, y_clamped)))
    
    # 如果有超出画面的点，找到距离有效点位最近的一个
    if out_of_bounds_points and valid_points:
        # 计算每个超出画面的点到所有有效点的最小距离
        min_distances = []
        for out_pt, out_dist, clamped_pt in out_of_bounds_points:
            min_dist_to_valid = float('inf')
            for valid_pt in valid_points:
                dist = np.sqrt((out_pt[0] - valid_pt[0])**2 + (out_pt[1] - valid_pt[1])**2)
                min_dist_to_valid = min(min_dist_to_valid, dist)
            min_distances.append((out_pt, out_dist, clamped_pt, min_dist_to_valid))
        
        # 找到距离有效点位最近的点
        closest_out_point = min(min_distances, key=lambda x: x[3])
        valid_points.append(closest_out_point[2])  # 添加限制到边界的点

    print(f"len(valid_points): {len(valid_points)}")
    print(f"len(out_of_bounds_points): {len(out_of_bounds_points)}")
    
    return valid_points

def cover_imgA_and_imgB(warped_A, img_B):
    gray = cv2.cvtColor(warped_A, cv2.COLOR_BGR2GRAY)
    # 创建掩码，非黑色区域为1，黑色区域为0
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.merge([mask, mask, mask])  # 转换为3通道
    
    alpha = 0.5  # 设置透明度
    result = img_B.copy().astype(np.float32)
    result = result * (1 - alpha * mask) + warped_A.astype(np.float32) * (alpha * mask)
    return result

# 归一化函数
def norm_pt(pt, img):
    h, w = img.shape[:2]
    return (round(pt[0] / w, 4), round(pt[1] / h, 4))


def main():
    # 1. 读取两张图片
    basedir = os.getcwd()
    print(f"basedir: {basedir}")
    path_A = os.path.join(basedir, 'python', 'base', 'base_highschool.png')
    path_B = os.path.join(basedir, 'python', 'testImgs', 'test_highschool', 'highschool_img.jpg')
    # path_A = os.path.join(basedir, 'python', 'base', 'base_standard.png')
    # path_B = os.path.join(basedir, 'python', 'testImgs', 'test_standard', 'tandard3.png')

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

    img_A = cv2.imread(path_A)
    img_B = cv2.imread(path_B)
    print(f"imgA.shape: {img_A.shape}")
    print(f"imgB.shape: {img_B.shape}")
    # imgA.shape: (1148, 1732, 3), w,h,3
    # imgB.shape: (1080, 1920, 3)
    
    if img_A is None or img_B is None:
        print("错误：无法读取图片，请检查路径是否正确")
        return

    
    # 2. 在A图上选择点（归一化）
    print(f"请在图片A上按顺序点击{NUM_POINTS}个点（图像已缩放显示）")
    points_line = select_points(img_A, "Select Points on Image A", scale=0.6)
    points_line_norm = [norm_pt(pt, img_A) for pt in points_line]
    print("points_line:", points_line)
    print("points_line_norm:", points_line_norm)
    
    # 归一化后的A图关键点（highschool）
    # 预设的6个对应打点
    points_A = [norm_pt(pt, img_A) for pt in [(201, 90), (633, 88), (1033, 83), (1475, 78), (640, 666), (1034, 668)]]
    # # 线1： 
    # highschool_line_one = [norm_pt(pt, img_A) for pt in [(55, 86), (53, 291), (51, 545), (55, 761), (55, 935), (56, 1123)]]
    # # 线2： 
    # highschool_line_two = [norm_pt(pt, img_A) for pt in [(56, 83), (201, 85), (640, 83), (1031, 80), (1468, 76), (1625, 78)]]
    # # 线3： 
    # highschool_line_three = [norm_pt(pt, img_A) for pt in [(1626, 75), (1625, 193), (1630, 348), (1630, 501), (1631, 688), (1630, 911)]]
    # # 三分线：
    # highschool_curve_threeline = [norm_pt(pt, img_A) for pt in [(200, 83), (200, 211), (213, 330), (245, 466), (290, 563), (341, 638), (406, 698), (473, 753), (551, 798), (655, 843), (761, 861), (918, 866), (1073, 828), (1201, 753), (1291, 681), (1360, 595), (1408, 515), (1441, 443), (1463, 325), (1471, 71)]]
    # # 罚球线：
    # highschool_rect_freethrow = [norm_pt(pt, img_A) for pt in [(636, 88), (641, 175), (640, 250), (638, 341), (635, 416), (638, 501), (638, 581), (640, 660), (730, 666), (825, 668), (925, 671), (1031, 663), (1033, 573), (1030, 508), (1033, 436), (1031, 351), (1035, 271), (1031, 200), (1031, 141), (1028, 78)]]
    # 罚篮圈：
    # highschool_free_throw_circle = [norm_pt(pt, img_A) for pt in [(641, 670), (650, 725), (671, 776), (711, 820), (761, 855), (833, 870), (903, 856), (960, 828), (996, 793), (1020, 750), (1031, 710), (1031, 663)]]
    # print("highschool_line_one:", highschool_line_one)
    # print("highschool_line_two:", highschool_line_two)
    # print("highschool_line_three:", highschool_line_three)
    # print("highschool_curve_threeline:", highschool_curve_threeline)
    # print("highschool_rect_freethrow:", highschool_rect_freethrow)
    """
highschool_line_one: [(0.0318, 0.0749), (0.0306, 0.2535), (0.0294, 0.4747), (0.0318, 0.6629), (0.0318, 0.8145), (0.0323, 0.9782)]
highschool_line_two: [(0.0323, 0.0723), (0.1161, 0.074), (0.3695, 0.0723), (0.5953, 0.0697), (0.8476, 0.0662), (0.9382, 0.0679)]
highschool_line_three: [(0.9388, 0.0653), (0.9382, 0.1681), (0.9411, 0.3031), (0.9411, 0.4364), (0.9417, 0.5993), (0.9411, 0.7936)]
highschool_curve_threeline: [(0.1155, 0.0723), (0.1155, 0.1838), (0.123, 0.2875), (0.1415, 0.4059), (0.1674, 0.4904), (0.1969, 0.5557), (0.2344, 0.608), (0.2731, 0.6559), (0.3181, 0.6951), (0.3782, 0.7343), (0.4394, 0.75), (0.53, 0.7544), (0.6195, 0.7213), (0.6934, 0.6559), (0.7454, 0.5932), (0.7852, 0.5183), (0.8129, 0.4486), (0.832, 0.3859), (0.8447, 0.2831), (0.8493, 0.0618)]
highschool_rect_freethrow: [(0.3672, 0.0767), (0.3701, 0.1524), (0.3695, 0.2178), (0.3684, 0.297), (0.3666, 0.3624), (0.3684, 0.4364), (0.3684, 0.5061), (0.3695, 0.5749), (0.4215, 0.5801), (0.4763, 0.5819), (0.5341, 0.5845), (0.5953, 0.5775), (0.5964, 0.4991), (0.5947, 0.4425), (0.5964, 0.3798), (0.5953, 0.3057), (0.5976, 0.2361), (0.5953, 0.1742), (0.5953, 0.1228), (0.5935, 0.0679)]
    """

    # 3. 在B图上选择点（归一化）
    print(f"请在图片B上按相同顺序点击{NUM_POINTS}个点（图像已缩放显示）")
    # points_B = select_points(img_B, "Select Points on Image B", scale)
    points_B = [norm_pt(pt, img_B) for pt in [(851, 428), (1030, 460), (1303, 495), (1753, 561), (623, 520), (860, 586)]]

    # points_B = [norm_pt(pt, img_B) for pt in [(0.09842352, 0.544697), (0.3336174, 0.5212121), (0.56923735, 0.5090909), (0.8150831, 0.46439394), (0.5717938, 0.79318184), (0.707712, 0.58257574)]]

    # standard：
    # points_B = [(43, 991), (923, 1006), (1631, 1015), (2068, 1026), (1620, 1203), (2378, 1178)]

    # print(f"选择的A图点坐标（原始分辨率、归一化后）: {points_A}")
    # print(f"选择的B图点坐标（原始分辨率、归一化后）: {points_B}")
    """
选择的A图点坐标（原始分辨率、归一化后）: [(0.1161, 0.0784), (0.3655, 0.0767), (0.5964, 0.0723), (0.8516, 0.0679), (0.3695, 0.5801), (0.597, 0.5819)]
选择的B图点坐标（原始分辨率、归一化后）: [(0.4432, 0.3963), (0.5365, 0.4259), (0.6786, 0.4583), (0.913, 0.5194), (0.3245, 0.4815), (0.4479, 0.5426)]
    """

    # 【工程端计算】4. 计算单应性矩阵（使用原始分辨率的坐标）
    # M为归一化矩阵,
    # 归一化坐标转像素坐标的变换
    # pts_A, pts_B为归一化坐标，需转为像素坐标（同时转为numpy数组），再计算单应性
    h_A, w_A = img_A.shape[:2]
    h_B, w_B = img_B.shape[:2]
    pts_A_px = np.array([[pt[0] * w_A, pt[1] * h_A] for pt in points_A], dtype=np.float32)
    pts_B_px = np.array([[pt[0] * w_B, pt[1] * h_B] for pt in points_B], dtype=np.float32)
    # print(f"pts_A_px: {pts_A_px}")
    # print(f"pts_B_px: {pts_B_px}")
    # # 计算像素坐标下的单应性矩阵
    M = findHomography(pts_A_px, pts_B_px)
    np.set_printoptions(precision=4, suppress=True) # 设置打印精度，log
    # print("M:", M)
    
    # 5. 应用透视变换    
    warped_A = cv2.warpPerspective(img_A, M, (img_B.shape[1], img_B.shape[0])) #全图映射，用于演示，cpp时关闭

    # 实际操作：将img_A上的第一个点映射到warped_A上
    # highschool_name = ['highschool_line_one', 'highschool_line_two', 'highschool_line_three', 'highschool_curve_threeline', 'highschool_rect_freethrow']
    # highschool_all_lines = [highschool_line_one, highschool_line_two, highschool_line_three, highschool_curve_threeline, highschool_rect_freethrow]
    # warped_all_lines = []
    # for i, line in enumerate(highschool_all_lines):
    #     warped_line_points = []
    #     print(f"\noutput {highschool_name[i]}:")
    #     for pt in line:
    #         # imgA的采样点为归一化坐标，需先转为imgA的像素坐标，再进行映射计算
    #         mapped_pt_px = (pt[0] * w_A, pt[1] * h_A)
    #         mapped_pt = map_point_A_to_warpedA(mapped_pt_px, M)
    #         warped_line_points.append(mapped_pt)
    #         print(f"{mapped_pt[0]}, {mapped_pt[1]}")
    #     warped_all_lines.append(warped_line_points)    

    # 6. 创建掩码，只叠加非黑色区域
    result = cover_imgA_and_imgB(warped_A, img_B)
    

    # 将mapped_pt点画到result图像上
    # for line in warped_all_lines:
    #     for pt in line:
    #         mapped_pt_int = (int(round(pt[0])), int(round(pt[1])))
    #         cv2.circle(result, mapped_pt_int, 10, (0, 0, 255), -1)  # 红色圆点，半径10
            # cv2.putText(result, f"{mapped_pt_int}", (mapped_pt_int[0]+10, mapped_pt_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # 单独打印三分线 - 使用边界过滤功能
    # 将归一化坐标转换为像素坐标
    points_line_px = [(pt[0] * w_A, pt[1] * h_A) for pt in points_line_norm]
    
    # 使用边界过滤功能，保留所有画面内的点和距离有效点最近的超出画面点
    warped_line_points = map_points_with_boundary_filter(points_line_px, M, img_B.shape)
    
    # 绘制映射后的点
    for mapped_pt in warped_line_points:
        mapped_pt_int = (int(round(mapped_pt[0])), int(round(mapped_pt[1])))
        cv2.circle(result, mapped_pt_int, 10, (0, 0, 255), -1)
    
    print("warped_line_points (边界过滤后):", warped_line_points)
    print("warped_line_points_norm:", [norm_pt(pt, img_B) for pt in warped_line_points])

    # 单独绘制罚球线
    special_line1 = [
        (636, 88), (641, 175),
        (640, 250), (638, 341),
        (635, 416), (638, 501),
        (638, 581), (640, 660)
    ]

    # 直接使用整型像素坐标，不做归一化
    special_line2 = [
        (1031.0, 663.0),
        (1033.0, 573.0),
        (1030.0, 508.0),
        (1033.0, 436.0),
        (1031.0, 351.0),
        (1035.0, 271.0),
        (1031.0, 200.0),
        (1031.0, 141.0),
        (1028.0, 78.0)
    ]

    special_line = [
        (640.0, 660.0), 
        (730.0, 666.0), (825.0, 668.0), 
        (925.0, 671.0), (1031.0, 663.0)
    ]

    print("special_line:", special_line)
    for pt in special_line:
        mapped_pt_px = (int(round(pt[0])), int(round(pt[1])))
        mapped_pt = map_point_A_to_warpedA(mapped_pt_px, M)
        mapped_pt_int = (int(round(mapped_pt[0])), int(round(mapped_pt[1])))
        cv2.circle(result, mapped_pt_int, 10, (0, 0, 255), -1)



    outdir = "orizip/res2"

    # 显示并保存结果
    result = result.astype(np.uint8)
    # 显示前缩小result，但保存原图
    display_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)  # 按0.5缩放显示
    cv2.imshow("Result", display_result)
    # 确保输出目录存在，若不存在则创建
    save_dir = os.path.join(basedir, outdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'result.jpg')
    # cv2.imwrite(save_path, result)
    # print(f"结果已保存为 {save_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()