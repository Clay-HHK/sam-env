# 快速开始：图像增强 + SAM 分割项目

本项目实现了一个完整的低光照图像增强 + SAM分割的处理流程，适用于图像前处理、边界提取、语义分割等任务的预处理阶段。包含增强、分割、后处理、轮廓绘制等模块，支持单图批处理、迭代分析和调试。

### 本人环境配置（仅供参考）

- 系统：Ubuntu 22.04
- CUDA：12.8
- Python：3.10
- GPU：NVIDIA Tesla T4 × 4
- Conda：24.9.2
- PyTorch：2.x（支持 GPU 加速）

### 环境配置

使用 [Anaconda](https://www.anaconda.com/) 管理环境：

```bash
conda env create -f environment.yml
conda activate sam-env
```

> ⚠️ 注意：本项目基于 Python 3.10 版本构建，部分依赖可能与更高版本不兼容。请严格按照 `environment.yml` 构建环境。

### 项目结构

```bash
sam-env/
├── src/
│   ├── enhance_image.py             # 图像增强工具函数库，封装了 CLAHE、Gamma 校正、亮度拉伸、锐化、饱和度提升等增 强方法（前期图像增强用）。
│   ├── process_image_full.py        # （可用，直接集成所有）标准增强 + SAM 分割流程。适用于测试完整流程效果（无反馈优化），输出增强图、掩码和轮廓图。推荐作为入门流程
│   ├── process_image_hybrid.py      # （可用，直接集成所有）混合增强方案测试脚本，结合多种图像增强策略，用于评估不同组合对分割效果的影响（如 CLAHE + gamma + 亮度拉伸 + RGB 引导等）
│   ├── process_image_improved.py    # （可用，直接集成所有）改进版处理流程，集成图像增强 + SAM 分割 + 自定义后处理（如面积筛选、mask 合并等），适合生产环境部署前测试        
│   └── segment_with_sam.py          # SAM 的掩码预测与评分接口，包括多掩码输出、评分筛选、调试保存等功能（前期debug图片用）
├── segment_anything/                # Meta AI 的 SAM 库
├── results/                         # 输出增强图、掩码图、轮廓图等
├── LOLdataset/                      # 本地测试用图像路径（可自定义）
├── sam_vit_h_4b8939.pth             # SAM 模型权重文件（需单独下载）
├── environment.yml                  # Conda 环境配置
└── README.md
```

### 图像增强 + 分割流程（主流程）

运行主脚本 `process_image_full.py` ,以此类推三个脚本都可以完整运行：

```bash
cd src
python process_image_full.py
```

输出图像保存在 `results/` 文件夹中，包括：

- `enhanced.png`：增强后图像
- `mask.png`：SAM分割掩码
- `contours.png`：轮廓绘制图
- `debug_masks/`：多掩码评分对比

### 模型权重（SAM）

需手动下载 `sam_vit_h_4b8939.pth` 并放在项目根目录下。模型来自 [Meta AI SAM GitHub](https://github.com/facebookresearch/segment-anything)。

### 数据集来源

可使用 LOL-Image Enhancement Dataset 中的 `our485/low/` 图像用于低光照测试。图像路径默认读取 `LOLdataset/our485/low/9.png`，可自行替换或批量处理。
