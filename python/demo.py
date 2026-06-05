import cv2
from ImageHomography import ImageHomographyProcessor
import sys
import os

def ask_path(prompt, default=None):
    p = input(f"{prompt} [{default}]: ").strip()
    return p or default

def main():
    print("将依次加载源图像(A)与模板图像(B)，在每张图上按顺序点击对应点（至少4个），中键或按任意键结束选择窗口显示。")
    default_a = "python/testImgs/left.png"
    # default_a = "python/testImgs/right.png"
    # default_b = "python/testImgs/test_standard/standard-dbb.png"
    default_b = "python/base/base_standard_annotated.png"
    path_a = ask_path("源图像A路径", default_a)
    path_b = ask_path("模板图像B路径", default_b)
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print("错误：文件不存在")
        sys.exit(1)

    img_A = cv2.imread(path_a)
    img_B = cv2.imread(path_b)
    print(f"加载图像A: {path_a} (shape: {img_A.shape if img_A is not None else '无法读取'})")
    print(f"加载图像B: {path_b} (shape: {img_B.shape if img_B is not None else '无法读取'})")
    if img_A is None or img_B is None:
        print("错误：无法读取图像")
        sys.exit(1)

    proc = ImageHomographyProcessor()

    num_points = input("要选择的点数量 (默认6): ").strip()
    try:
        num_points = int(num_points) if num_points else 6
    except:
        num_points = 6

    pts_A_pixel = [(1163, 594), (962, 615), (716, 639), (363, 679), (1292, 683), (991, 743)]
    pts_B_pixel = [(40, 4), (199, 5), (386, 4), (543, 2), (199, 223), (384, 224)]
    if num_points > len(pts_A_pixel):
        print(f"预设点最多只有 {len(pts_A_pixel)} 个，已自动使用全部预设点")
        num_points = len(pts_A_pixel)
    if num_points < 4:
        print("需要至少4个点")
        sys.exit(1)

    # 预设点坐标，方便调试。实际使用时可改为交互式选择。
    pts_A = pts_A_pixel[:num_points]
    pts_B = pts_B_pixel[:num_points]
    print(f"使用像素坐标点 A: {pts_A}")
    print(f"使用像素坐标点 B: {pts_B}")
    if False:
        pts_A = proc.select_points(img_A, "Select Points on Image A", scale=1.0)
        print(pts_A)
        if len(pts_A) < 4:
            print("需要至少4个点")
            sys.exit(1)
        pts_B = proc.select_points(img_B, "Select Points on Image B", scale=1.0)
        print(pts_B)
        if len(pts_B) < 4:
            print("需要至少4个点")
            sys.exit(1)
    
    # 转为 numpy float32 像素坐标
    import numpy as np
    ptsA_px = np.array(pts_A, dtype=np.float32)
    ptsB_px = np.array(pts_B, dtype=np.float32)
    
    # 归一化到 [0, 1] 范围进行计算（提高数值稳定性）
    ptsA_norm = np.array([[x / img_A.shape[1], y / img_A.shape[0]] for x, y in ptsA_px], dtype=np.float32)
    ptsB_norm = np.array([[x / img_B.shape[1], y / img_B.shape[0]] for x, y in ptsB_px], dtype=np.float32)
    print(f"归一化坐标点 A: {ptsA_norm}")
    print(f"归一化坐标点 B: {ptsB_norm}")
    H_norm = proc.find_homography(ptsA_norm, ptsB_norm)
    print(f"计算得到的归一化坐标单应性矩阵 H_norm:\n{H_norm}")
    # 测试点映射
    testPtA = (0.5, 0.5)
    mappedPt = proc.map_point_A_to_warpedA(testPtA, H_norm)
    print(f"测试点 {testPtA} 映射到B后的坐标: {mappedPt}")

    # H_norm: A归一化坐标 -> B归一化坐标
    # warpPerspective 需要 A像素坐标 -> B像素坐标：
    #   pB_px = S_B * H_norm * S_A^-1 * pA_px
    S_A_inv = np.array([[1.0 / img_A.shape[1], 0.0, 0.0], [0.0, 1.0 / img_A.shape[0], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    S_B = np.array([[img_B.shape[1], 0.0, 0.0], [0.0, img_B.shape[0], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    H = S_B @ H_norm @ S_A_inv
    H = H / H[2, 2]
    print(f"转换为像素空间后用于 warpPerspective 的矩阵 H:\n{H}")

    ptsA_h = np.column_stack([ptsA_px, np.ones(len(ptsA_px), dtype=np.float32)])
    reproj = (H @ ptsA_h.T).T
    reproj = reproj[:, :2] / reproj[:, 2:3]
    errors = np.linalg.norm(reproj - ptsB_px, axis=1)
    print(f"A点映射到B后的坐标:\n{np.round(reproj, 2)}")
    print(f"对应点重投影误差(px): {np.round(errors, 3)}, mean={errors.mean():.3f}, max={errors.max():.3f}")

    warped_A = cv2.warpPerspective(img_A, H, (img_B.shape[1], img_B.shape[0]))
    result = proc.cover_imgA_and_imgB(warped_A, img_B, alpha=0.7)
    cv2.imshow("Warped Overlay", cv2.resize(result, (0,0), fx=0.6, fy=0.6))
    k = cv2.waitKey(0)
    if k == ord('s'):
        out = "homography_quick_result.png"
        cv2.imwrite(out, result)
        print(f"已保存: {out}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
