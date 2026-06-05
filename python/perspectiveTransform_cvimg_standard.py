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
NUM_POINTS = 40 # 固定需要选择的点数

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
    path_A = os.path.join(basedir, 'python', 'base', 'base_standard.png')
    path_B = os.path.join(basedir, 'python', 'testImgs', 'test_standard', 'standard3.png')

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
    
    if img_A is None or img_B is None:
        print("错误：无法读取图片，请检查路径是否正确")
        return

    # 预设的6个对应打点
    points_A = [norm_pt(pt, img_A) for pt in [(41, 3), (196, 1), (383, 3), (543, 5), (200, 223), (383, 225)]]

    
    # 2. 在A图上选择点（归一化）
    print(f"请在图片A上按顺序点击{NUM_POINTS}个点（图像已缩放显示）")
    points_line = select_points(img_A, "Select Points on Image A", scale=0.6)
    points_line_norm = [norm_pt(pt, img_A) for pt in points_line]
    print("points_line:", points_line)
    print("points_line_norm:", points_line_norm)
    

    # 3. 在B图上选择点（归一化）
    print(f"请在图片B上按相同顺序点击{NUM_POINTS}个点（图像已缩放显示）")
    # points_B = select_points(img_B, "Select Points on Image B", scale)
    # standard：
    points_B = [norm_pt(pt, img_B) for pt in [(43, 991), (923, 1006), (1631, 1015), (2068, 1026), (1620, 1203), (2378, 1178)]]

    # print(f"选择的A图点坐标（原始分辨率、归一化后）: {points_A}")
    # print(f"选择的B图点坐标（原始分辨率、归一化后）: {points_B}")


    # 【工程端计算】4. 计算单应性矩阵（使用原始分辨率的坐标）
    # M为归一化矩阵,
    # 归一化坐标转像素坐标的变换
    # pts_A, pts_B为归一化坐标，需转为像素坐标（同时转为numpy数组），再计算单应性
    h_A, w_A = img_A.shape[:2]
    h_B, w_B = img_B.shape[:2]
    pts_A_px = np.array([[pt[0] * w_A, pt[1] * h_A] for pt in points_A], dtype=np.float32)
    pts_B_px = np.array([[pt[0] * w_B, pt[1] * h_B] for pt in points_B], dtype=np.float32)
    M = findHomography(pts_A_px, pts_B_px)
    np.set_printoptions(precision=4, suppress=True)
    warped_A = cv2.warpPerspective(img_A, M, (img_B.shape[1], img_B.shape[0]))
    result = cover_imgA_and_imgB(warped_A, img_B)


    '''
    # 归一化后的A图关键点（standard）    
    # # 线1： 
    # highschool_line_one = [norm_pt(pt, img_A) for pt in []]
    points_line: [(1, 323), (1, 260), (1, 201), (3, 153), (1, 103), (1, 50), (1, 1)]
    points_line_norm: [(0.0017, 0.9362), (0.0017, 0.7536), (0.0017, 0.5826), (0.0052, 0.4435), (0.0017, 0.2986), (0.0017, 0.1449), (0.0017, 0.0029)]

    # # 线2： 
    # highschool_line_two = []
    points_line: [(1, 1), (41, 1), (96, 5), (156, 5), (205, 3), (265, 3), (330, 5), (403, 1), (476, 5), (530, 3), (573, 3)]
    points_line_norm: [(0.0017, 0.0029), (0.0706, 0.0029), (0.1652, 0.0145), (0.2685, 0.0145), (0.3528, 0.0087), (0.4561, 0.0087), (0.568, 0.0145), (0.6936, 0.0029), (0.8193, 0.0145), (0.9122, 0.0087), (0.9862, 0.0087)]

    # # 线3： 
    # highschool_line_three = [norm_pt(pt, img_A) for pt in []]
    points_line: [(576, 3), (576, 70), (578, 136), (578, 205), (576, 265), (576, 321)]
    points_line_norm: [(0.9914, 0.0087), (0.9914, 0.2029), (0.9948, 0.3942), (0.9948, 0.5942), (0.9914, 0.7681), (0.9914, 0.9304)]

    # # 三分线：
    # highschool_curve_threeline = [norm_pt(pt, img_A) for pt in []]
points_line: [(38, 3), (36, 51), (36, 113), (50, 163), (73, 203), (101, 241), (135, 270), (175, 293), (213, 310), (260, 320), (295, 323), (331, 318), (370, 308), (405, 295), (433, 278), (465, 255), (481, 233), (505, 208), (523, 178), (538, 143), (545, 118), (543, 66), (543, 25), (541, 3)]
points_line_norm: [(0.0654, 0.0087), (0.062, 0.1478), (0.062, 0.3275), (0.0861, 0.4725), (0.1256, 0.5884), (0.1738, 0.6986), (0.2324, 0.7826), (0.3012, 0.8493), (0.3666, 0.8986), (0.4475, 0.9275), (0.5077, 0.9362), (0.5697, 0.9217), (0.6368, 0.8928), (0.6971, 0.8551), (0.7453, 0.8058), (0.8003, 0.7391), (0.8279, 0.6754), (0.8692, 0.6029), (0.9002, 0.5159), (0.926, 0.4145), (0.938, 0.342), (0.9346, 0.1913), (0.9346, 0.0725), (0.9312, 0.0087)]

# 占位线5
points_line: [(198, 3), (198, 50), (198, 101), (196, 165), (196, 223)]
points_line_norm: [(0.3408, 0.0087), (0.3408, 0.1449), (0.3408, 0.2928), (0.3373, 0.4783), (0.3373, 0.6464)]

# 占位线6
points_line: [(385, 225), (383, 175), (383, 128), (385, 70), (383, 3)]
points_line_norm: [(0.6627, 0.6522), (0.6592, 0.5072), (0.6592, 0.371), (0.6627, 0.2029), (0.6592, 0.0087)]

    # # 罚球线：
    # highschool_rect_freethrow = [norm_pt(pt, img_A) for pt in []]
points_line: [(198, 225), (223, 225), (263, 226), (313, 225), (356, 226), (385, 226)]
points_line_norm: [(0.3408, 0.6522), (0.3838, 0.6522), (0.4527, 0.6551), (0.5387, 0.6522), (0.6127, 0.6551), (0.6627, 0.6551)]

    # 罚篮圈：
    # highschool_free_throw_circle = [norm_pt(pt, img_A) for pt in []]
points_line: [(223, 225), (233, 261), (253, 283), (291, 295), (328, 285), (351, 258), (358, 225)]
points_line_norm: [(0.3838, 0.6522), (0.401, 0.7565), (0.4355, 0.8203), (0.5009, 0.8551), (0.5645, 0.8261), (0.6041, 0.7478), (0.6162, 0.6522)]


# 过滤
points_line: [(1, 311), (1, 248), (1, 160), (1, 71), (0, 1), (86, 1), (176, 3), (271, 3), (361, 1), (443, 3), (511, 3), (576, 3), (576, 75), (576, 155), (576, 230), (576, 310)]
points_line_norm: [(0.0017, 0.9014), (0.0017, 0.7188), (0.0017, 0.4638), (0.0017, 0.2058), (0.0, 0.0029), (0.148, 0.0029), (0.3029, 0.0087), (0.4664, 0.0087), (0.6213, 0.0029), (0.7625, 0.0087), (0.8795, 0.0087), (0.9914, 0.0087), (0.9914, 0.2174), (0.9914, 0.4493), (0.9914, 0.6667), (0.9914, 0.8986)]
    '''

    # 实际操作：将img_A上的第一个点映射到warped_A上
    # standard_name = ['standard_line_one', 'standard_line_two', 'standard_line_three', 'standard_curve_threeline', 'standard_rect_freethrow']
    # standard_all_lines = [standard_line_one, standard_line_two, standard_line_three, standard_curve_threeline, standard_rect_freethrow]
    # warped_all_lines = []
    # for i, line in enumerate(highschool_all_lines):
    #     warped_line_points = []
    #     print(f"\noutput {standard_name[i]}:")
    #     for pt in line:
    #         # imgA的采样点为归一化坐标，需先转为imgA的像素坐标，再进行映射计算
    #         mapped_pt_px = (pt[0] * w_A, pt[1] * h_A)
    #         mapped_pt = map_point_A_to_warpedA(mapped_pt_px, M)
    #         warped_line_points.append(mapped_pt)
    #         print(f"{mapped_pt[0]}, {mapped_pt[1]}")
    #     warped_all_lines.append(warped_line_points)    
   

    # 将mapped_pt点画到result图像上
    # for line in warped_all_lines:
    #     for pt in line:
    #         mapped_pt_int = (int(round(pt[0])), int(round(pt[1])))
    #         cv2.circle(result, mapped_pt_int, 10, (0, 0, 255), -1)  # 红色圆点，半径10
            # cv2.putText(result, f"{mapped_pt_int}", (mapped_pt_int[0]+10, mapped_pt_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # 单独打印三分线
    warped_line_points = []
    for pt in points_line_norm:
        mapped_pt_px = (pt[0] * w_A, pt[1] * h_A)
        mapped_pt = map_point_A_to_warpedA(mapped_pt_px, M)
        mapped_pt_int = (int(round(mapped_pt[0])), int(round(mapped_pt[1])))
        cv2.circle(result, mapped_pt_int, 10, (0, 0, 255), -1)
        warped_line_points.append(mapped_pt)
    print("warped_line_points:", warped_line_points)
    print("warped_line_points_norm:", [norm_pt(pt, img_B) for pt in warped_line_points])




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