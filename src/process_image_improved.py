import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

def get_center_points(img, num_points=5):
    """获取图像中心区域的多个提示点"""
    h, w = img.shape[:2]
    points = []
    
    # 中心点
    points.append([w//2, h//2])
    
    # 四个象限的中心点
    points.append([w//4, h//4])
    points.append([3*w//4, h//4])
    points.append([w//4, 3*h//4])
    points.append([3*w//4, 3*h//4])
    
    return np.array(points[:num_points])

def segment_with_prompts(img, predictor, results_dir, debug_dir):
    """使用提示点进行分割"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    # 获取提示点
    input_points = get_center_points(img)
    input_labels = np.ones(len(input_points))  # 1表示前景点
    
    # 使用提示点分割
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # 选择最佳掩码
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    print(f"提示点分割 - 最佳得分: {best_score:.3f}")
    
    # 保存调试信息
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_uint = (mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(debug_dir, f"prompt_mask_{i}_s{score:.3f}.png"), mask_uint)
    
    # 在图像上标记提示点
    debug_img = img.copy()
    for point in input_points:
        cv2.circle(debug_img, tuple(point), 5, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(debug_dir, "prompt_points.png"), debug_img)
    
    return (best_mask.astype(np.uint8) * 255), best_score


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


def main():
    # 路径配置
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    low_path = os.path.join(base_dir, "LOLdataset/our485/low/25.png")
    checkpoint = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
    results_dir = os.path.join(base_dir, "results")
    debug_dir = os.path.join(results_dir, "debug_improved")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # 读取图像
    img = cv2.imread(low_path)
    if img is None:
        raise FileNotFoundError(f"找不到输入图像: {low_path}")
    
    # 温和增强
    enhanced = enhance_image_v2(img, use_sharpen=True, sat_boost=True, scale=1.3)
    cv2.imwrite(os.path.join(results_dir, "enhanced_improved.png"), enhanced)
    
    # 加载SAM
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to("cuda").eval()
    predictor = SamPredictor(sam)
    
    # 提示点分割
    mask, score = segment_with_prompts(enhanced, predictor, results_dir, debug_dir)
    
    # 后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 减小核大小
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cv2.imwrite(os.path.join(results_dir, "mask_improved.png"), mask_clean)
    
    # 轮廓提取
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = enhanced.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(results_dir, "contours_improved.png"), contour_img)
    
    print(f"[√] 改进版分割完成，得分: {score:.3f}")

if __name__ == "__main__":
    main()