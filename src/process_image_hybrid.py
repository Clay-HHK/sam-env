import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

def get_distributed_points(img, num_points=9):
    """获取分布更均匀的提示点，包括边缘区域"""
    h, w = img.shape[:2]
    points = []
    
    # 网格分布的点，确保覆盖整个图像
    for i in range(3):
        for j in range(3):
            x = int(w * (j + 1) / 4)
            y = int(h * (i + 1) / 4)
            points.append([x, y])
    
    return np.array(points[:num_points])

def enhance_image_v2(img, use_sharpen=False, sat_boost=False, scale=1.2):
    # === 1. CLAHE 增强 ===
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # === 2. Gamma 校正 ===
    def gamma_correction(image, gamma=1.6):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    enhanced = gamma_correction(enhanced, gamma=1.6)

    # === 3. 可选锐化（Unsharp Mask）===
    if use_sharpen:
        def unsharp_mask(image, alpha=1.3, beta=-0.3, ksize=(5, 5)):
            image = image.astype(np.float32)
            blurred = cv2.GaussianBlur(image, ksize, 0)
            sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        enhanced = unsharp_mask(enhanced)

    # === 4. 可选饱和增强 ===
    if sat_boost:
        def boost_saturation(image, boost_value=20):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, boost_value)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        enhanced = boost_saturation(enhanced)

    # === 5. 线性亮度提升 ===
    def brighten_linear(image, scale=1.2):
        return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    
    enhanced = brighten_linear(enhanced, scale=scale)

    return enhanced

def hybrid_segment(img, predictor, results_dir, debug_dir):
    """混合分割方法：结合提示点分割和自动分割"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    # 方法1：使用分布式提示点
    input_points = get_distributed_points(img)
    input_labels = np.ones(len(input_points))  # 1表示前景点
    
    masks_prompt, scores_prompt, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # 方法2：自动分割
    masks_auto, scores_auto, _ = predictor.predict(multimask_output=True)
    
    # 合并所有候选掩码
    all_masks = np.concatenate([masks_prompt, masks_auto])
    all_scores = np.concatenate([scores_prompt, scores_auto])
    
    # 选择最佳掩码
    best_idx = -1
    best_score = -1.0
    best_area = 0
    
    for i, (mask, score) in enumerate(zip(all_masks, all_scores)):
        mask_uint = mask.astype(np.uint8) * 255
        area = cv2.countNonZero(mask_uint)
        
        # 保存调试信息
        method = "prompt" if i < len(masks_prompt) else "auto"
        cv2.imwrite(os.path.join(debug_dir, f"{method}_mask_{i}_s{score:.3f}_a{area}.png"), mask_uint)
        
        print(f"{method.capitalize()} Mask {i}: score={score:.3f}, area={area}")
        
        # 选择标准：优先考虑得分，但也要有合理的面积
        if area > 1000 and score > best_score:  # 最小面积阈值
            best_score = score
            best_idx = i
            best_area = area
    
    if best_idx < 0:
        raise RuntimeError("未找到有效掩码")
    
    best_mask = all_masks[best_idx]
    
    # 在图像上标记提示点
    debug_img = img.copy()
    for point in input_points:
        cv2.circle(debug_img, tuple(point), 5, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(debug_dir, "distributed_points.png"), debug_img)
    
    print(f"[√] 最佳掩码: 得分={best_score:.3f}, 面积={best_area}")
    
    return (best_mask.astype(np.uint8) * 255), best_score

def adaptive_postprocess(mask, results_dir):
    """自适应后处理：根据掩码特征调整参数"""
    # 分析掩码特征
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    # 计算最大轮廓的特征
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    
    # 根据面积和周长比例选择核大小
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if compactness < 0.3:  # 细长形状，使用较小的核
            kernel_size = 5
        else:  # 紧凑形状，使用中等大小的核
            kernel_size = 9
    else:
        kernel_size = 7
    
    print(f"自适应后处理: 核大小={kernel_size}")
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先闭运算连接断开的部分
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 再开运算去除噪声
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 保守的连通组件过滤：只移除非常小的区域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    
    # 计算面积阈值（保留面积大于总面积1%的组件）
    total_area = mask.shape[0] * mask.shape[1]
    area_threshold = max(100, total_area * 0.01)  # 最小100像素
    
    clean_mask = np.zeros_like(opened)
    for i in range(1, num):  # 跳过背景（标签0）
        if stats[i, cv2.CC_STAT_AREA] > area_threshold:
            clean_mask[labels == i] = 255
    
    cv2.imwrite(os.path.join(results_dir, "mask_clean.png"), clean_mask)
    
    return clean_mask

def main():
    # 路径配置
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    low_path = os.path.join(base_dir, "LOLdataset/our485/low/9.png")
    checkpoint = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
    results_dir = os.path.join(base_dir, "results")
    debug_dir = os.path.join(results_dir, "debug_hybrid")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(low_path)
    if img is None:
        raise FileNotFoundError(f"找不到输入图像: {low_path}")
    
    # 图像增强
    enhanced = enhance_image_v2(img, use_sharpen=True, sat_boost=True, scale=1.3)
    cv2.imwrite(os.path.join(results_dir, "enhanced_hybrid.png"), enhanced)
    
    # 加载SAM
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to("cuda").eval()
    predictor = SamPredictor(sam)
    
    # 混合分割
    mask, score = hybrid_segment(enhanced, predictor, results_dir, debug_dir)
    cv2.imwrite(os.path.join(results_dir, "mask_hybrid_raw.png"), mask)
    
    # 自适应后处理
    mask_clean = adaptive_postprocess(mask, results_dir)
    
    # 轮廓提取
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = enhanced.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(results_dir, "contours_hybrid.png"), contour_img)
    
    print(f"[√] 混合分割完成，最终得分: {score:.3f}")
    print(f"[√] 检测到 {len(contours)} 个轮廓")

if __name__ == "__main__":
    main()