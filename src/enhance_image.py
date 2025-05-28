import cv2
import numpy as np
import os

input_path = "/home/p1/data/hhk/sam-env/LOLdataset/our485/low/2.png" 
output_path = "/home/p1/data/hhk/sam-env/results/enhanced.png"
os.makedirs("results", exist_ok=True)

# === 读取图像 ===
img = cv2.imread(input_path)

# === CLAHE 增强 ===
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
merged = cv2.merge((cl, a, b))
enhanced_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# === Gamma 校正 ===
def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

enhanced_img = gamma_correction(enhanced_clahe, gamma=1.5)

# === 保存结果图 ===
cv2.imwrite(output_path, enhanced_img)
print(f"[√] 图像增强完成，已保存至：{output_path}")