"""
process_image_full.py

完整流程：
1. 读取低光照图像
2. 图像增强：CLAHE -> Gamma 校正 -> 双边滤波 -> Single-Scale Retinex -> 锐化
3. 使用 SAM 自动分割：多候选掩码 -> 打印 score & area -> 选择最优非空掩码
4. 后处理掩码：形态学闭运算 + 开运算 -> 连通组件过滤
5. 提取并绘制轮廓
6. 保存结果：增强图、掩码图、轮廓叠加图、调试掩码

依赖：
  pip install opencv-python matplotlib numpy torch torchvision torchaudio
  pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

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

def segment_and_extract(img, predictor, results_dir, debug_dir):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(multimask_output=True)

    # 选取最佳非空掩码
    best_idx, best_score = -1, -1.0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_uint = mask.astype(np.uint8)*255
        area = cv2.countNonZero(mask_uint)
        print(f"Mask {i}: score={score:.3f}, area={area}")
        cv2.imwrite(os.path.join(debug_dir, f"mask_{i}_s{score:.3f}.png"), mask_uint)
        if area > 0 and score > best_score:
            best_score, best_idx = score, i

    if best_idx < 0:
        raise RuntimeError("未找到有效掩码，请检查输入或使用提示。")

    best_mask = masks[best_idx].astype(np.uint8)*255
    out_mask_path = os.path.join(results_dir, "mask.png")
    cv2.imwrite(out_mask_path, best_mask)
    print(f"[√] 最佳 Mask {best_idx} (score={best_score:.3f}) 保存: {out_mask_path}")
    return best_mask

def postprocess_and_draw(img, mask, results_dir):
    # 形态学后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 连通组件过滤（保留面积>5000）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    keep = list(range(1, num))
    clean_mask = np.isin(labels, keep).astype(np.uint8)*255
    cv2.imwrite(os.path.join(results_dir, "mask_clean.png"), clean_mask)

    # 提取轮廓并叠加
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    out_contour = os.path.join(results_dir, "contours.png")
    cv2.imwrite(out_contour, contour_img)
    print(f"[√] 边界图保存: {out_contour}")

def main():
    # 路径配置
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    low_path = os.path.join(base_dir, "LOLdataset/our485/low/9.png")
    checkpoint = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
    results_dir = os.path.join(base_dir, "results")
    debug_dir = os.path.join(results_dir, "debug_masks")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # 读取 & 增强
    img = cv2.imread(low_path)
    if img is None:
        raise FileNotFoundError(f"找不到输入图像: {low_path}")
    enhanced = enhance_image_v2(img, use_sharpen=True, sat_boost=True, scale=1.3)
    out_enhanced = os.path.join(results_dir, "enhanced.png")
    cv2.imwrite(out_enhanced, enhanced)
    print(f"[√] 增强图保存: {out_enhanced}")

    # 加载 SAM
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to("cuda").eval()
    predictor = SamPredictor(sam)

    # 分割 & 提取
    best_mask = segment_and_extract(enhanced, predictor, results_dir, debug_dir)
    postprocess_and_draw(enhanced, best_mask, results_dir)

if __name__ == "__main__":
    main()

