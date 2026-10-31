# PetVision

基于分割辅助分类的自然场景宠物图像理解系统。

## 研究问题

项目研究：在自然场景中，先提取宠物前景，再进行品种分类，是否能够减少复杂背景对分类结果的干扰？

系统包含：

- 宠物前景分割；
- Oxford-IIIT Pet 37 类品种分类；
- 原图分类与分割辅助分类对照；
- Top-5 概率、mask、叠加图和前景提取可视化；
- Gradio 真实模型推理界面。

当前实现采用二值前景 mask。Zenodo 标签包提供紫色背景和黄色前景的二值分割图；程序将其转换为 `1=foreground`、`2=background`。镜像中缺少 mask 的 4 个图像会从 trainval/test 列表中过滤，当前可用记录为 7348 条。报告中应明确这一数据处理事实，不将其描述为原始三值 trimap。

## 环境安装

在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Windows NVIDIA GPU users must replace the CPU PyTorch wheel with the official
CUDA wheel:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu128
```

也可以直接使用仓库中的 CUDA 依赖清单：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-cuda-cu128.txt
```

Verify the device before training:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If the first output is `False`, do not start a formal GPU experiment. CPU is
appropriate only for data checks, code validation, and small smoke runs.

## 数据准备

```powershell
.venv\Scripts\python.exe scripts\download_dataset.py --keep-archives
.venv\Scripts\python.exe scripts\check_dataset.py
.venv\Scripts\python.exe scripts\validate_dataset.py --limit 0
.venv\Scripts\python.exe scripts\visualize_samples.py --count 6
```

默认数据目录：`data/oxford-iiit-pet/`。

数据来源：

- 图像镜像：Hugging Face `Shiym/ViT-FineTune` 的固定 revision；
- 二值分割标签：Zenodo record `7116591` 的 `oxford_pets_labels.zip`；
- trainval/test 列表：固定 Hugging Face revision 中的列表文件；
- 原始数据说明：<https://www.robots.ox.ac.uk/~vgg/data/pets/>。

程序优先尝试官方 Oxford 下载地址；当前网络环境下如果官方 TLS 下载失败，会自动使用上述镜像。

## 训练

### 1. 分割模型

```powershell
.venv\Scripts\python.exe scripts\train_segmentation.py `
  --epochs 20 `
  --batch-size 8 `
  --image-size 256 `
  --device auto `
  --output artifacts\segmentation
```

RTX 4060 Laptop users should start with `--batch-size 2` for segmentation.
If CUDA memory is insufficient, use `--batch-size 1`; keep `image-size 256`
for the final comparison unless the experiment explicitly studies resolution.

输出：

- `artifacts/segmentation/best.pt`；
- `artifacts/segmentation/history.json`；
- `artifacts/segmentation/dataset.json`。

### 2. 原图分类基线

```powershell
.venv\Scripts\python.exe scripts\train_classification.py `
  --mode raw `
  --epochs 20 `
  --batch-size 16 `
  --image-size 256 `
  --device auto `
  --output artifacts\classification\raw
```

### 3. 分割辅助分类

先完成分割模型训练，再执行：

```powershell
.venv\Scripts\python.exe scripts\train_classification.py `
  --mode guided `
  --segmenter-checkpoint artifacts\segmentation\best.pt `
  --epochs 20 `
  --batch-size 16 `
  --image-size 256 `
  --device auto `
  --output artifacts\classification\guided
```

分割辅助图像使用软 mask：

```text
x_guided = x * (alpha + (1 - alpha) * foreground_probability)
```

默认 `alpha=0.25`，可通过 `--alpha` 调整。

### 小规模 smoke run

不应将 smoke 结果写入报告最终结果。它只用于检查代码链路：

```powershell
.venv\Scripts\python.exe scripts\train_segmentation.py `
  --epochs 1 --limit 8 --image-size 64 --batch-size 2 `
  --output artifacts\smoke_segmentation --device cpu

.venv\Scripts\python.exe scripts\train_classification.py `
  --mode raw --epochs 1 --limit 8 --image-size 64 --batch-size 2 `
  --output artifacts\smoke_raw --device cpu

.venv\Scripts\python.exe scripts\train_classification.py `
  --mode guided `
  --segmenter-checkpoint artifacts\smoke_segmentation\best.pt `
  --epochs 1 --limit 8 --image-size 64 --batch-size 2 `
  --output artifacts\smoke_guided --device cpu
```

## 测试集评价

```powershell
.venv\Scripts\python.exe scripts\evaluate.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --output artifacts\evaluation `
  --device auto
```

评价包含：

- 分割 foreground IoU、mIoU、Dice；
- 原图分类 Accuracy、Top-5 Accuracy、Macro Precision、Macro Recall、Macro-F1；
- 预测 mask 辅助分类；
- 真实 mask oracle 对照。

`true_mask_oracle` 只用于估计理想上限，不能作为实际部署流程。

## 背景鲁棒性实验

```powershell
.venv\Scripts\python.exe scripts\evaluate_background.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --output artifacts\robustness `
  --device auto
```

程序比较原始背景、背景模糊和灰色背景下的原图分类与分割辅助分类结果。

## 案例和训练曲线

```powershell
.venv\Scripts\python.exe scripts\export_examples.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --output artifacts\examples `
  --count 24 `
  --device auto

.venv\Scripts\python.exe scripts\plot_history.py `
  --history artifacts\segmentation\history.json `
  --kind segmentation `
  --output artifacts\segmentation\history.png

.venv\Scripts\python.exe scripts\plot_history.py `
  --history artifacts\classification\raw\history.json `
  --kind classification `
  --output artifacts\classification\raw\history.png
```

## 启动系统

```powershell
.venv\Scripts\python.exe scripts\app.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --device auto
```

浏览器打开 `http://127.0.0.1:7860`。

## 程序结构

```text
petvision/
├── config.py                 # 共享配置
├── data/
│   ├── oxford_pet.py         # 数据索引和 mask 转换
│   ├── dataset.py            # PyTorch Dataset
│   ├── splits.py             # 分层训练/验证划分
│   └── transforms.py         # 图像与 mask 同步预处理
├── models/
│   ├── classifier.py         # 自定义 CNN
│   └── unet.py               # U-Net
├── evaluation/metrics.py     # 分类和分割指标
├── training/loops.py         # BP 训练循环和软 mask
└── inference.py              # 单图完整推理
```

## 课程关联

- CNN：卷积层、ReLU、池化和层级视觉特征提取；
- BP：分类交叉熵、分割交叉熵/Dice 损失的反向传播；
- HMAX 启发：讨论局部特征、层级特征和目标选择思想；
- 类脑视觉：将前景 mask 视为一种简化的选择性注意机制。

Hopfield、LSTM 和 TopoICA 与本项目主任务没有直接必要关系，不强行加入系统。

## License and data

The source code is released under the MIT License in `LICENSE`.

The dataset, downloaded archives, generated masks, checkpoints, logs, and
experiment outputs are not included in this repository. They are ignored by
Git and must be obtained or generated locally. Oxford-IIIT Pet and the
third-party mirrors have their own terms; users must review those terms before
redistributing data or trained weights.

## 验证命令

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q petvision scripts tests
```
