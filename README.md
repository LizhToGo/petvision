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
- 验证集选择权重的原图/分割辅助分类晚期融合；

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

### AMD ROCm

For AMD Instinct or Radeon GPUs, do not install the NVIDIA CUDA wheel above.
Use a ModelScope or Linux image with a preinstalled ROCm-compatible PyTorch
build. Install the remaining dependencies without replacing PyTorch:

```bash
python -m pip install -r requirements-rocm.txt
python -m pip install -e . --no-deps
```

Verify the ROCm device:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The training code uses standard PyTorch operators and the `torch.cuda` device
API, which is also the compatibility API exposed by ROCm builds. The exact
ROCm wheel must match the cloud image, Python version, and driver; do not
reuse `requirements-cuda-cu128.txt` on an AMD instance.

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

### 4. ResNet-18 迁移学习对照

以下实验在现有划分和数据预处理上，分别训练原图与分割辅助分类器。
两者均从 torchvision 的 ImageNet-1K V1 预训练权重初始化，并微调整个
ResNet-18；分类头针对 37 类重新初始化。首次运行需要下载预训练权重，
之后加载本实验的 checkpoint 不需要再次下载。ROCm 用户需要同时安装与
已安装 PyTorch 匹配的 torchvision 版本。

```powershell
.venv\Scripts\python.exe scripts\train_classification.py `
  --architecture resnet18 --pretrained --mode raw `
  --epochs 20 --batch-size 4 --image-size 256 --learning-rate 0.0001 `
  --seed 42 --device auto --output artifacts\experiments\resnet18_imagenet\raw

.venv\Scripts\python.exe scripts\train_classification.py `
  --architecture resnet18 --pretrained --mode guided `
  --segmenter-checkpoint artifacts\segmentation\best.pt `
  --epochs 20 --batch-size 4 --image-size 256 --learning-rate 0.0001 `
  --seed 42 --device auto --output artifacts\experiments\resnet18_imagenet\guided

.venv\Scripts\python.exe scripts\evaluate_fusion.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\experiments\resnet18_imagenet\raw\best.pt `
  --guided-checkpoint artifacts\experiments\resnet18_imagenet\guided\best.pt `
  --output artifacts\experiments\resnet18_imagenet\fusion `
  --batch-size 4 --lambda-step 0.05 --seed 42 --device auto
```

两条新分支采用相同的设置与分割引导系数，仅比较原图和预测 mask 辅助输入。
每个分支按验证集 Macro-F1 保存最佳模型，融合权重也只在验证集选择。
评估输出在 `fusion/metrics.json`；新的模型应使用其中验证集选出的
`metadata.raw_weight` 启动演示，不能沿用旧实验的 0.23。
相较于从零训练的 SmallCNN，ResNet-18 同时改变了架构与预训练，不能将性能
差异单独归因于网络结构；单个随机种子的结果也不能说明统计显著性。

云端 Linux 可使用同样的固定配置顺序运行两组训练及融合评价，例如：

```bash
export DATA_ROOT="$PWD/data/oxford-iiit-pet"
export SEGMENTATION_CHECKPOINT="$PWD/artifacts/segmentation/best.pt"
export RUN_DIR="$PWD/artifacts/experiments/resnet18_cloud_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/artifacts/experiments"
mkdir "$RUN_DIR"
nohup bash scripts/run_resnet18_cloud.sh >"$RUN_DIR/train.log" 2>&1 &
echo "PID=$!  RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/train.log"
```

根据实际环境修改数据和分割权重路径。脚本要求 GPU 可用，并拒绝覆盖已有
实验输出；中断后如需续训，使用对应 `last.pt` 调用下文的 `--resume`。

### 5. 两阶段微调与 early stopping

上一轮结果显示 ResNet-18 的训练损失继续下降，但验证集 Macro-F1 在第
4～5 轮后回落。因此可以冻结 backbone 训练分类头，再用更小的 backbone
学习率解冻微调：

```powershell
.venv\Scripts\python.exe scripts\train_classification.py `
  --architecture resnet18 --pretrained --schedule staged --mode raw `
  --head-epochs 2 --epochs 15 `
  --backbone-learning-rate 0.00001 --head-learning-rate 0.0001 `
  --early-stopping-patience 4 --batch-size 4 --image-size 256 `
  --seed 42 --device auto `
  --output artifacts\experiments\resnet18_staged\raw
```

guided 分支使用相同参数并增加 `--mode guided` 和
`--segmenter-checkpoint`。`best.pt` 按验证集 Macro-F1 保存；该实验的
目的不是增加训练轮数，而是检验分阶段优化能否减轻迁移学习过拟合。

云端大显存实例可以在固定 `256×256` 输入下提高 batch 和数据加载并行度：
`scripts/run_resnet18_staged_cloud.sh` 默认使用 `BATCH_SIZE=32`、
`NUM_WORKERS=8`、30 个 epoch 上限和 `--fast`。显存或 CPU 不足时可在
启动前设置 `BATCH_SIZE=16`、`NUM_WORKERS=4`；`--fast` 允许固定输入尺寸的
GPU 算子和 benchmark，正式报告应记录该设置。

### Training loop meaning

For each batch, the program loads and preprocesses images, runs a forward
pass, computes the loss, calls `loss.backward()`, and updates parameters with
AdamW. After each epoch it evaluates the validation split without gradients
and saves the checkpoint with the best validation metric. The segmentation
loss combines cross-entropy and Dice loss; classification uses cross-entropy.
The raw classifier is the baseline, while guided classification first uses a
frozen U-Net to produce a soft foreground mask.

### 断点续训

每个 epoch 结束后，训练脚本会保存 `last.pt`，其中包含模型参数、
optimizer 状态、当前 epoch、最佳指标和 history。正式任务中断后，可用：

```powershell
.venv\Scripts\python.exe scripts\train_segmentation.py `
  --resume artifacts\segmentation\last.pt `
  --epochs 20 --batch-size 8 --image-size 256 --device cuda `
  --output artifacts\segmentation

.venv\Scripts\python.exe scripts\train_classification.py `
  --mode raw `
  --resume artifacts\classification\raw\last.pt `
  --epochs 20 --batch-size 16 --image-size 256 --device cuda `
  --output artifacts\classification\raw
```

`--epochs` is the final epoch number, not the number of additional epochs.
An older `best.pt` without optimizer state can still be loaded as a warm start,
but it is not an exact optimizer-level resume.

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

## Raw/guided late fusion

This experiment does not retrain either classifier. It searches the raw
classifier probability weight on the validation split, then evaluates the
selected weight once on the test split:

```powershell
.venv\Scripts\python.exe scripts\evaluate_fusion.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --output artifacts\fusion `
  --device auto
```

The fusion probability is:

```text
p_fusion = lambda * p_raw + (1 - lambda) * p_guided
```

The test set is not used to select `lambda`.

## 演示中的晚期融合

当前正式实验选择 `raw=0.23`、`guided=0.77`。启动界面时会同时展示
原图分类、分割辅助分类和晚期融合分类的 Top-5 结果：

```powershell
.venv\Scripts\python.exe scripts\app.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --fusion-raw-weight 0.23 `
  --device auto
```

如果重新训练了模型，应先运行 `scripts/evaluate_fusion.py`，再将验证集
选择出的 `raw_weight` 传给 `scripts/app.py`；不要直接使用测试集调出的权重。

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
  --fusion-raw-weight 0.23 `
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
