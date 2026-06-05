import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import signal  # 用于捕获 Ctrl+C 信号
# from util.cal_new import CoordTransformer 
# from util.utils import getAngleDict, scale_points
# 移除不必要的全局导入（原代码中的CoordTransformer未使用，保留注释说明）
# os.path.append('./')
# with open('./python/testvideo/test1/NumMatch.json', 'r') as f:
#     num_match = json.load(f)

class ImageHomographyProcessor:
    """图像单应性变换处理器，封装所有状态和操作，消除全局变量"""

    def __init__(self):
        # 实例变量存储状态，替代全局变量
        self.current_points = None
        self.current_image = None
        self.current_scale = 1.0
        self.window_name = ""
        self.angle_M = {}

    def _click_event(self, event, x, y, flags, param):
        """鼠标点击事件处理方法（内部使用）"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 将显示坐标转换为原始图像坐标
            original_x = int(x / self.current_scale)
            original_y = int(y / self.current_scale)

            # 记录原始坐标（使用实例变量）
            self.current_points.append((original_x, original_y))

            # 在显示图像上画圆标记（使用显示坐标）
            cv2.circle(self.current_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                self.current_image,
                str(len(self.current_points)),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(self.window_name, self.current_image)

        if event == cv2.EVENT_MBUTTONDOWN:
            cv2.destroyWindow(self.window_name)

    def select_points(self, image, title, scale=0.5):
        """
        在图像上交互式选择点并返回原始图像坐标
        :param image: 原始图像（numpy数组）
        :param title: 窗口标题
        :param scale: 图像显示缩放比例
        :return: 选中的点列表 [(x1,y1), (x2,y2), ...]
        """
        # 初始化实例状态
        self.current_points = []
        self.current_scale = scale
        self.window_name = title

        # 缩放图像用于显示
        height, width = image.shape[:2]
        display_width = int(width * scale)
        display_height = int(height * scale)
        self.current_image = cv2.resize(image, (display_width, display_height))

        # 设置鼠标回调（通过partial传递self）
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._click_event)

        # 显示图像并等待用户操作
        cv2.imshow(self.window_name, self.current_image)
        cv2.waitKey(0)

        # 清理窗口
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(self.window_name)

        # 返回选中的点（原始坐标）
        return self.current_points

    @staticmethod
    def find_homography(src_pts, dst_pts):
        """
        使用最小二乘法（SVD）计算单应性矩阵
        :param src_pts: 源点列表（至少4个）
        :param dst_pts: 目标点列表（数量与源点一致）
        :return: 3x3单应性矩阵
        """
        assert len(src_pts) == len(dst_pts) >= 4, "源点和目标点数量必须相等且至少4个"

        # 构建矩阵A
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

        A = np.array(A)

        # SVD分解求解
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        # 归一化（保证H[2,2]=1）
        H = H / H[2, 2]
        return H

    @staticmethod
    def map_point_A_to_warpedA(pt, M):
        """
        将源图像A的点通过单应性矩阵映射到变换后的图像上
        :param pt: (x, y) 源点坐标
        :param M: 3x3单应性矩阵
        :return: (x', y') 映射后的坐标（浮点数）
        """
        pt_homo = np.array([[pt[0], pt[1], 1]], dtype=np.float32).T  # 齐次坐标 3x1
        mapped = np.dot(M, pt_homo)
        mapped /= mapped[2, 0]  # 归一化
        return (mapped[0, 0], mapped[1, 0])

    @staticmethod
    def map_points_with_boundary_filter(points, M, img_B_shape):
        """
        映射点列表，保留画面内的有效点+距离最近的越界点（限制到边界）
        :param points: 源点列表
        :param M: 单应性矩阵
        :param img_B_shape: 目标图像尺寸 (height, width)
        :return: 过滤后的有效点列表
        """
        height, width = img_B_shape[:2]
        valid_points = []  # 画面内的点
        out_of_bounds_points = []  # 越界点

        # 映射所有点并分类
        for pt in points:
            mapped_pt = ImageHomographyProcessor.map_point_A_to_warpedA(pt, M)
            x, y = mapped_pt

            if 0 <= x < width and 0 <= y < height:
                valid_points.append(mapped_pt)
            else:
                # 计算越界点到画面边界的最近点
                x_clamped = max(0, min(x, width - 1))
                y_clamped = max(0, min(y, height - 1))
                distance = np.sqrt((x - x_clamped) ** 2 + (y - y_clamped) ** 2)
                out_of_bounds_points.append((mapped_pt, distance, (x_clamped, y_clamped)))

        # 若有越界点，添加距离有效点最近的越界点（限制到边界）
        if out_of_bounds_points and valid_points:
            min_distances = []
            for out_pt, out_dist, clamped_pt in out_of_bounds_points:
                min_dist_to_valid = float("inf")
                for valid_pt in valid_points:
                    dist = np.sqrt((out_pt[0] - valid_pt[0]) ** 2 + (out_pt[1] - valid_pt[1]) ** 2)
                    min_dist_to_valid = min(min_dist_to_valid, dist)
                min_distances.append((out_pt, out_dist, clamped_pt, min_dist_to_valid))

            # 取距离有效点最近的越界点
            closest_out_point = min(min_distances, key=lambda x: x[3])
            valid_points.append(closest_out_point[2])

        print(f"有效点数量: {len(valid_points)}")
        print(f"越界点数量: {len(out_of_bounds_points)}")

        return valid_points

    @staticmethod
    def cover_imgA_and_imgB(warped_A, img_B, alpha=0.6):
        """
        将变换后的img_A叠加到img_B上（仅叠加非黑色区域）
        :param warped_A: 透视变换后的img_A
        :param img_B: 目标图像B
        :param alpha: 叠加透明度（0-1）
        :return: 叠加后的图像
        """
        # 生成掩码：非黑色区域为1，黑色区域为0
        gray = cv2.cvtColor(warped_A, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        mask = cv2.merge([mask, mask, mask])  # 转为3通道

        # 混合图像
        result = img_B.copy().astype(np.float32)
        result = result * (1 - alpha * mask) + warped_A.astype(np.float32) * (alpha * mask)
        return result.astype(np.uint8)

    @staticmethod
    def norm_pt(pt, img):
        """
        将像素坐标归一化到[0,1]范围
        :param pt: (x, y) 像素坐标
        :param img: 原始图像
        :return: (nx, ny) 归一化坐标
        """
        h, w = img.shape[:2]
        return (round(pt[0] / w, 4), round(pt[1] / h, 4))

    @staticmethod
    def remove_out_of_bounds_points(points_A, points_B, keep_num=None):
        for i in range(len(points_A)-1, -1, -1):
            x, y = points_B[i]
            x2, y2 = points_A[i]
            if x <= 0.0 or y <= 0.0 or x >= 1.0 or y >= 1.0 or x2 <= 0.0 or y2 <= 0.0 or x2 >= 1.0 or y2 >= 1.0:
                # print(f"移除越界点: A点{points_A[i]} -> B点{points_B[i]}")
                del points_A[i]
                del points_B[i]
    
    def run(self, yaw_angle, path_A, path_B, points_A=None, points_B=None, num_points=None, player_point=None, show=True):
        """
        主执行流程：读取图像→选择点→计算单应性矩阵→透视变换→叠加图像
        :param path_A: 源图像A的路径, 
        :param path_B: 目标图像B的路径, 模板图片
        :param points_A: 预定义的图像A的点（可选，无需交互选择）
        :param points_B: 预定义的图像B的点（可选，无需交互选择）
        :param num_points: 要求选择的点数量（仅交互选择时生效）
        """
        # 1. 读取图像
        if isinstance(path_A, str):
            img_A = cv2.imread(path_A)
        else:
            img_A = path_A
        if isinstance(path_B, str):
            img_B = cv2.imread(path_B)
        else:
            img_B = path_B

        # 校验图像读取
        if img_A is None:
            print(f"错误：无法读取图像A，请检查路径: {path_A}")
            return
        if img_B is None:
            print(f"错误：无法读取图像B，请检查路径: {path_B}")
            return

        # print(f"图像A尺寸: {img_A.shape}")
        # print(f"图像B尺寸: {img_B.shape}")

        # 2. 交互式选择点（若未提供预定义点）
        if points_A is None:
            assert num_points is not None, "未提供points_A时，必须指定num_points"
            print(f"\n请在图像A上按顺序点击{num_points}个点（图像已缩放显示）")
            points_A = self.select_points(img_A, "Select Points on Image A", scale=2.0)
            print(f"图像A选中的点: {points_A}")
            # 校验点数量
            if len(points_A) != num_points:
                print(f"警告：选择的点数量({len(points_A)})与要求的({num_points})不一致")

        if points_B is None:
            assert num_points is not None, "未提供points_B时，必须指定num_points"
            print(f"\n请在图像B上按顺序点击{num_points}个点（图像已缩放显示）")
            points_B = self.select_points(img_B, "Select Points on Image B", scale=1.0)
            print(f"图像B选中的点: {points_B}")
            # 校验点数量
            if len(points_B) != num_points:
                print(f"警告：选择的点数量({len(points_B)})与要求的({num_points})不一致")


        # 3. 归一化点坐标
        points_A_norm = [self.norm_pt(pt, img_A) for pt in points_A]
        points_B_norm = [self.norm_pt(pt, img_B) for pt in points_B]
        # print(f"图像A归一化点: {points_A_norm}")
        # print(f"图像B归一化点: {points_B_norm}")

        # 移除越界点
        self.remove_out_of_bounds_points(points_A_norm, points_B_norm)
        # print(f"图像A的点: {points_A_norm}")
        # print(f"图像B的点: {points_B_norm}")

        # 4. 转换回像素坐标（用于计算单应性矩阵）
        h_A, w_A = img_A.shape[:2]
        h_B, w_B = img_B.shape[:2]
        pts_A_px = np.array([[pt[0] * w_A, pt[1] * h_A] for pt in points_A_norm], dtype=np.float32)
        pts_B_px = np.array([[pt[0] * w_B, pt[1] * h_B] for pt in points_B_norm], dtype=np.float32)

        # 5. 计算单应性矩阵
        if self.angle_M.get(yaw_angle) is not None:
            self.M = self.angle_M[yaw_angle]
        else:
            self.M = self.find_homography(pts_A_px, pts_B_px)
            self.angle_M[yaw_angle] = self.M
        # for point in points_A:
            # mapped_point = self.map_point_A_to_warpedA(point, self.M)
            # print(f"图像A点 {point} 映射到变换后图像坐标: {np.round(mapped_point, 2)}")
        np.set_printoptions(precision=4, suppress=True)
        # print(f"\n单应性矩阵M:\n{self.M}")

        # 6. 应用透视变换
        warped_A = cv2.warpPerspective(img_A, self.M, (img_B.shape[1], img_B.shape[0]))
        player_point_warpped = {}
        if player_point is not None:
            for trk_id, pt in player_point.items():
                mapped_pt = self.map_point_A_to_warpedA(pt, self.M)
                player_point_warpped[trk_id] = mapped_pt
        # # 仿射变换
        # self.M = cv2.getAffineTransform(pts_A_px[:3], pts_B_px[:3])

        # np.set_printoptions(precision=4, suppress=True)
        # print(f"\n仿射变换矩阵M:\n{self.M}")
        # warped_A = cv2.warpAffine(img_A, self.M[:2], (img_B.shape[1], img_B.shape[0]))
        # 7. 图像叠加（修复原代码参数传反的问题）
        result = self.cover_imgA_and_imgB(warped_A, img_B)
        if player_point is not None:
            for trk_id,pt in player_point_warpped.items():
                cv2.circle(result, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
                cv2.putText(result, str(trk_id) + '-' + str(num_match.get(str(trk_id), '')), (int(pt[0])+10, int(pt[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # 8. 显示和保存结果
        if show:
            display_result = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)  # 缩放显示
            cv2.imshow("Result", display_result)
            if cv2.waitKey(0) == 27:  # 按下 ESC 键退出
                cv2.destroyAllWindows()
                exit()
            cv2.destroyAllWindows()

        # 可选：保存结果
        # cv2.imwrite("homography_result.png", result)
        return result

# def transform_coordinates_between_angles(user_points, src_yaw, src_pitch, angle_info):
#     """
#     将用户指定的坐标点从源角度（yaw/pitch）映射到目标角度的虚拟坐标，
#     再转换回实际坐标。

#     参数:
#         user_points (list): 用户选择的实际坐标点列表，格式为 [(x1, y1), (x2, y2), ...]
#         src_yaw (float): 源偏航角（原始角度）
#         src_pitch (float): 源俯仰角（原始角度）
#         angle_info (tuple): 目标角度信息，格式为 (yaw, pitch, X, Y, W, H)

#     返回:
#         list: 转换后的实际坐标点列表
#     """
#     tf = CoordTransformer()
#     user_virtual_point_list = []
#     for (x, y) in user_points:
#         vpt = tf.real2vir((x, y), src_yaw, src_pitch)
#         user_virtual_point_list.append(vpt)
#     rpt_list = []
#     for point in user_virtual_point_list:
#         rpt = list(tf.vir2real(point, angle_info[0], angle_info[1]))
#         rpt_list.append(rpt)
#     rpt_list = scale_points(rpt_list, angle_info[2:6])  # 反变换缩放
#         # print(f"Yaw {dst_yaw}: Real: ({point[0]},{point[1]}) -> Vir: ({rpt[0]:.2f},{rpt[1]:.2f})")
    
#     return rpt_list

def generate_point(width=1920, height=1080, num_points=40, show=False):
    # 用于标定
    # 对角线 1 
    x_coor = np.linspace(0, width, num=num_points) + np.random.randint(-50, 50, size=num_points)
    y_coor = np.linspace(0, height, num=num_points) + np.random.randint(-50, 50, size=num_points)
    points1 = np.column_stack((x_coor, y_coor)).astype(np.float32)
    # 对角线 2
    x_coor = np.linspace(0, width, num=num_points) + np.random.randint(-50, 50, size=num_points)
    y_coor = np.linspace(height, 0, num=num_points) + np.random.randint(-50, 50, size=num_points)
    points2 = np.column_stack((x_coor, y_coor)).astype(np.float32)

    # 水平线
    y_coor = np.linspace(0, height, num=num_points) + np.random.randint(-50, 50, size=num_points)
    x_coor = np.random.randint(0, width, size=num_points)
    points3 = np.column_stack((x_coor, y_coor)).astype(np.float32)

    # 垂直线
    x_coor = np.linspace(0, width, num=num_points) + np.random.randint(-50, 50, size=num_points)
    y_coor = np.random.randint(0, height, size=num_points)
    points4 = np.column_stack((x_coor, y_coor)).astype(np.float32)

    points = np.vstack((points1, points2, points3, points4))
    points = points[(points[:,0] >= 10) & (points[:,0] <= width - 10) & (points[:,1] >= 10) & (points[:,1] <= height - 10)]
    # 生成一张空白图像并绘制点
    if show:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, (x, y) in enumerate(points):
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(img, str(idx+1), (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Generated Points", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points.tolist()

interrupted = False

def signal_handler(sig, frame):
    """捕获 Ctrl+C 信号的回调函数"""
    global interrupted
    interrupted = True
    print("\n检测到 Ctrl+C 中断，正在保存视频...")
    
# def main(data):
#     # 注册 Ctrl+C 信号处理器
#     global interrupted
#     signal.signal(signal.SIGINT, signal_handler)
#     if "each_player_coor_path" in data:
#         with open(data["each_player_coor_path"], 'r') as f:
#             player_coordinate_dict = json.load(f)
#     # 初始化单应性变换处理器
#     homography_processor = ImageHomographyProcessor()
#     template_image = cv2.imread(data["template_image_path"])

#     # 2. 计算单应性矩阵
#     homography_matrix_left = homography_processor.find_homography(data["user_calibration_points_left"], data["template_calibration_points_left"])
#     homography_matrix_right = homography_processor.find_homography(data["user_calibration_points_right"], data["template_calibration_points_right"])

#     # 3. 生成标定图像随机点
#     generated_points_on_calib_img = generate_point()

#     # 4. 映射点到模板坐标系
#     mapped_points_on_template_left = []
#     mapped_points_on_template_right = []
#     for single_point in generated_points_on_calib_img:
#         mapped_single_point_left = homography_processor.map_point_A_to_warpedA(single_point, homography_matrix_left)
#         mapped_single_point_right = homography_processor.map_point_A_to_warpedA(single_point, homography_matrix_right)
#         mapped_points_on_template_left.append(mapped_single_point_left)
#         mapped_points_on_template_right.append(mapped_single_point_right)

#     # 5. 读取角度字典和视频
#     angle_dict = getAngleDict(data["angle_dict_path"])
#     input_video_path = data["input_video_path"]
#     output_video_path = data["output_video_path"]

#     # 初始化视频捕获
#     video_capture = cv2.VideoCapture(input_video_path)
#     if not video_capture.isOpened():
#         print(f"错误：无法打开输入视频 {input_video_path}")
#         exit(1)

#     # 获取视频原始参数（避免分辨率/帧率不匹配）
#     fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30
#     total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     # print(f"视频信息：{template_image.shape[1]}x{template_image.shape[0]} | 帧率 {fps} | 总帧数 {total_frames}")

#     # 初始化视频写入器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (template_image.shape[1], template_image.shape[0]))

#     # 6. 处理视频帧
#     frame_id = -1
#     success_frames = 0
#     bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame", colour="green")
#     try:
#         while True:
#             # 终止条件：中断信号/达到最大帧数/读完所有帧
#             if interrupted:
#                 break

#             ret, frame = video_capture.read()
#             if not ret:
#                 print(f"警告：读取帧 {frame_id} 失败")
#                 break
            
#             # 校验角度数据
#             if frame_id not in angle_dict:
#                 print(f"警告：帧 {frame_id} 无角度数据，跳过")
#                 frame_id += 1
#                 continue
#             frame_id += 1
#             # 坐标转换 + 单应性变换
#             yaw_angle = angle_dict[frame_id][0]
#             try:
#                 if yaw_angle >= 0.0:
#                     transformed_points = transform_coordinates_between_angles(
#                         generated_points_on_calib_img, data["user_calibration_yaw_right"], 0.0, angle_dict[frame_id]
#                     )
#                     point_template = mapped_points_on_template_right
#                 else:
#                     transformed_points = transform_coordinates_between_angles(
#                         generated_points_on_calib_img, data["user_calibration_yaw_left"], 0.0, angle_dict[frame_id]
#                     )
#                     point_template = mapped_points_on_template_left
#                 # 计算player的脚点坐标 
#                 feet_point_dict = {}
#                 for trk_id, player_coor in player_coordinate_dict.get(str(frame_id)).items():
#                     feet_point_dict[trk_id] = ((player_coor[0] + player_coor[2]) // 2, player_coor[3])  # 脚点坐标 (x, y)
#                 canvas = homography_processor.run(
#                     yaw_angle=yaw_angle, path_A=frame, path_B=template_image,
#                     points_A=transformed_points, points_B=point_template, show=False,
#                     player_point=feet_point_dict,
#                 )
#                 # 左上角稍微下方些显示帧号 以及当前的 yaw/pitch
#                 cv2.putText(canvas, f"Frame: {frame_id}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
#                 _info = "".join([str(i) +' ' for i in  angle_dict[frame_id]])
#                 cv2.putText(canvas, _info, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

#                 # if cv2.waitKey(0) == 27:  # 按下 ESC 键退出
#                 #     interrupted = True
#                 # 校验处理后的帧
#                 if canvas is None or canvas.shape[:2] != (template_image.shape[0], template_image.shape[1]):
#                     print(canvas.shape)
#                     print(f"警告：帧 {frame_id} 处理无效，使用原帧")
#                     canvas = frame

#                 video_writer.write(canvas)
#                 bar.update(1)
#                 success_frames += 1

#             except Exception as e:
#                 import traceback
#                 print(f"错误：处理帧 {frame_id} 失败 - {str(e)}")
#                 traceback.print_exc()
#                 video_writer.write(frame)  # 出错时写入原帧


#     finally:
#         # 无论是否异常/中断，必执行资源释放
#         video_capture.release()
#         video_writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n处理结束！成功写入 {success_frames}/{frame_id} 帧")
#         print(f"输出视频路径：{output_video_path}")

#         # 验证视频文件有效性
#         if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1024:
#             print("✅ 视频文件保存成功，可正常播放")
#         else:
#             print("❌ 视频文件生成失败或为空")

# if __name__ == "__main__":
#     data ={
#         "input_video_path": "./python/testvideo/c54afc1c59a94b239af010661835ecf2.mp4",
#         "angle_dict_path": "python/testvideo/c54afc1c59a94b239af010661835ecf2/c54afc1c59a94b239af010661835ecf2.json",
#         "output_video_path": "./python/testvideo/c54afc1c59a94b239af010661835ecf2_homography_video.mp4",
#         "user_calibration_points_left": [(1059, 501), (882, 518), (599, 543), (219, 574), (1185, 520), (283, 624), (1163, 568), (866, 623)],
#         "user_calibration_points_right": [(500, 505), (535, 509), (725, 525), (995, 546), (1290, 570), (1370, 578), (400, 581), (750, 625)],
#         "template_image_path": "/Users/xiaodai/Code/XbotGo/XbotGo-Algorithm-ZoneMapping-main/python/base/base_NBA_full.png",
#         "template_calibration_points_left": [(67, 128), (67, 618), (68, 1189), (67, 1680), (420, 127), (420, 1681), (750, 618), (750, 1187)],
#         "template_calibration_points_right": [(3380, 18), (3380, 123), (3380, 616), (3380, 1188), (3380, 1677), (3380, 1788), (2695, 620), (2695, 1188)],
#         "user_calibration_yaw_left": -53.33,
#         "user_calibration_yaw_right": 58.09
#     }
#     data ={
#         "input_video_path": "./python/testvideo/test1.mp4",
#         "angle_dict_path": "python/testvideo/test1/test1.json",
#         "output_video_path": "./python/testvideo/test1_homography_video.mp4",
#         "user_calibration_points_left": [(1170, 624), (995, 643), (706, 673), (344, 711), (1312, 642), (380, 765), (1294, 698), (955, 755)],
#         "user_calibration_points_right": [(945, 602), (1120, 624), (1430, 664), (1835, 712), (820, 615), (1794, 766), (845, 665), (1180, 736)],
#         "template_image_path": "/Users/xiaodai/Code/XbotGo/XbotGo-Algorithm-ZoneMapping-main/python/base/base_NBA_full.png",
#         "template_calibration_points_left": [(67, 128), (67, 618), (68, 1189), (67, 1680), (420, 127), (420, 1681), (750, 618), (750, 1187)],
#         "template_calibration_points_right": [(3380, 125), (3380, 617), (3380, 1188), (3380, 1679), (3025, 124), (3025, 1677), (2693, 617), (2693, 1186)],
#         "each_player_coor_path": "./python/testvideo/test1/EachPersonRealPoint.json",
#         "user_calibration_yaw_left": -49.55,
#         "user_calibration_yaw_right": 52.38
#     }
#     main(data)
