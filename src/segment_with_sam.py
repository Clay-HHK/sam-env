# segment_with_sam.py

import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# —— 路径配置 —— 
base_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
enhanced_path   = os.path.join(base_dir, "results", "enhanced.png")
checkpoint_path = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
out_mask        = os.path.join(base_dir, "results", "mask.png")
out_contour     = os.path.join(base_dir, "results", "contours.png")
debug_dir       = os.path.join(base_dir, "results", "debug_masks")
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# —— 读取增强图像 —— 
img = cv2.imread(enhanced_path)
if img is None:
    raise FileNotFoundError(f"找不到增强图像：{enhanced_path}")

# —— 加载 SAM 模型 —— 
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to("cuda")
sam.eval()
predictor = SamPredictor(sam)

# —— 生成多候选掩码 —— 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predictor.set_image(img_rgb)
masks, scores, _ = predictor.predict(
    multimask_output=True
)

# —— 过滤空掩码 & 选最高 score —— 
best_idx   = -1
best_score = -1.0
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask_uint = (mask.astype(np.uint8) * 255)
    area = int(cv2.countNonZero(mask_uint))
    print(f"[Debug] Mask {i}: score={score:.3f}, area={area}")
    cv2.imwrite(os.path.join(debug_dir, f"mask_{i}_s{score:.3f}.png"), mask_uint)
    if area == 0:
        continue
    if score > best_score:
        best_score = score
        best_idx   = i

if best_idx < 0:
    raise RuntimeError("所有候选掩码均为空，请检查输入图像或使用提示模式。")

# —— 取最佳掩码并后处理 —— 
mask = (masks[best_idx].astype(np.uint8) * 255)

# —— 形态学操作：先闭合再开运算，去除噪点、填充孔洞 —— 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean  = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imwrite(out_mask, mask_clean)
print(f"[√] 最佳 Mask {best_idx} (score={best_score:.3f})，已保存：{out_mask}")

# —— 提取轮廓并叠加 —— 
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
cv2.imwrite(out_contour, contour_img)
print(f"[√] 边界叠加图已保存：{out_contour}")
