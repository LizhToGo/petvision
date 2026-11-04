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
默认 `AUGMENTATION=color`，对训练图像施加轻微亮度、对比度和饱和度
扰动；复现当前无增强基线时设置 `AUGMENTATION=none`。验证集与测试集
不使用增强。

### 6. ResNet-50 原图分类基线

下一轮只训练 raw 分类器，不重训 U-Net、不训练 guided 或融合头，也不自动
评价测试集。`resnet50 --pretrained` 使用 torchvision 的
`ResNet50_Weights.IMAGENET1K_V2`；ResNet-18 仍使用 V1，已有 checkpoint
的参数键和加载方式不变。这是模型与预训练方案的对照，不是纯网络深度消融。

正式配置：256×256、ColorJitter 与水平翻转、FP32、batch 32、AdamW
weight decay 0.01；前 4 轮冻结 backbone，之后解冻；backbone/head 学习率
分别为 1e-5/1e-4，最多 30 轮，验证 Macro-F1 连续 6 轮不改善时早停。
本轮不增加余弦衰减、TTA 或新的融合模块。

分类训练现在使用两个独立的种子：`--split-seed` 固定划分，`--seed`
只控制模型初始化、shuffle 与增强。两者新运行均默认为 42；续训时默认从
新 checkpoint 恢复。重复实验只改变 `--seed`，不要改变 `--split-seed`。
新的 `split.json` 和 checkpoint 保存具体训练/验证图像 ID；续训时若样本
列表不同会报错。旧分类 checkpoint 没有保存这些信息，续训时需明确提供
原有划分参数。现有评价脚本的 `--seed` 仍控制划分，应与分类训练的
`--split-seed` 一致。

从项目根目录启动，数据目录继续指向原有数据：

```bash
export DATA_ROOT=/mnt/workspace/petvision/data/oxford-iiit-pet
export RUN_DIR=/mnt/workspace/petvision-runs/resnet50-256-color-raw-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p /mnt/workspace/petvision-runs
mkdir "$RUN_DIR"

# 可选性能预跑：真实训练数据、全模型反向传播；不保存模型，不选择精度方案。
python -u -m scripts.benchmark_classification \
  --root "$DATA_ROOT" --output "$RUN_DIR/benchmark" \
  --architecture resnet50 --pretrained --image-size 256 \
  --batch-sizes 32 64 128 --precisions fp32 bf16 \
  --warmup 5 --steps 20 --num-workers 8

# 第一轮正式对照固定 batch=32、FP32，不根据预跑自动更改。
export BATCH_SIZE=32 NUM_WORKERS=8 IMAGE_SIZE=256
export EPOCHS=30 HEAD_EPOCHS=4 PATIENCE=6
export BACKBONE_LR=0.00001 HEAD_LR=0.0001 WEIGHT_DECAY=0.01
export SPLIT_SEED=42 TRAINING_SEED=42
export PRECISION=fp32
nohup bash scripts/run_resnet50_raw_cloud.sh >"$RUN_DIR/train.log" 2>&1 &
echo "PID=$!  RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/train.log"
```

预跑记录每组的吞吐、平均损失及峰值 allocated/reserved 显存。计时包含
数据加载、前向、反向和 AdamW，不包含验证、完整评价指标或 checkpoint
存盘；每组先预热并同步 GPU，再计时。不支持 BF16 或显存不足的组合会
明确标记，不会静默切换精度。预跑权重直接丢弃，正式训练重新加载预训练
权重。不得把本机 GPU 的预跑结果当作云端 GPU 的性能。

结果文件：`environment.json` 记录设备和软件版本；`raw/config.json`、
`raw/split.json`、`raw/dataset.json` 保存配置与划分；`raw/history.json`
记录训练/验证曲线、耗时和 GPU 峰值显存；`raw/best.pt` 按验证 Macro-F1
选择，`raw/last.pt` 用于续训。比较候选期间只使用验证集。

中断后不要重新运行拒绝覆盖的云端脚本，改用：

```bash
python -u -m scripts.train_classification \
  --root "$DATA_ROOT" --mode raw \
  --resume "$RUN_DIR/raw/last.pt" --output "$RUN_DIR/raw" \
  --epochs 30 --image-size 256 --batch-size 32 --num-workers 8 \
  --device cuda --fast
```

续训恢复模型、优化器和训练阶段；不承诺与未中断运行逐位一致。现有
`train_feature_fusion.py` 仍仅支持 ResNet-18 的冻结特征实验，不要将
ResNet-50 checkpoint 传给该入口。

### 7. BF16 正式训练对照

完成 FP32 基线后，下一轮只改变训练精度；batch 仍为 32，划分、增强、
学习率、冻结阶段和 early stopping 均保持一致。不要从上一轮 FP32
`best.pt` 续训来充当 BF16 对照；新实验重新从同一 ImageNet V2 权重初始化。
本轮不需要重复性能预跑，也不自动评价测试集。

分类入口新增 `--precision fp32|bf16`；云端脚本使用 `PRECISION`，未设置
时仍为 FP32。BF16 仅包裹分类器训练前向和损失计算，验证与原基线一样
使用 FP32。参数与 AdamW 状态保留 FP32，不将整个模型转为 BF16。
`config.json` 和 checkpoint 分别记录 `precision` 与 `validation_precision`；
日志开头应显示 `precision=bf16 validation_precision=fp32`。CPU 或不支持
原生 BF16 的 CUDA/ROCm 设备会明确报错，不会静默退回 FP32。

在更新后的代码目录中执行，使用新的实验目录：

```bash
export DATA_ROOT=/mnt/workspace/petvision/data/oxford-iiit-pet
export RUN_DIR=/mnt/workspace/petvision-runs/resnet50-256-color-raw-bf16-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p /mnt/workspace/petvision-runs
mkdir "$RUN_DIR"
export PRECISION=bf16 BATCH_SIZE=32 NUM_WORKERS=8 IMAGE_SIZE=256
export EPOCHS=30 HEAD_EPOCHS=4 PATIENCE=6
export BACKBONE_LR=0.00001 HEAD_LR=0.0001 WEIGHT_DECAY=0.01
export SPLIT_SEED=42 TRAINING_SEED=42
nohup bash scripts/run_resnet50_raw_cloud.sh >"$RUN_DIR/train.log" 2>&1 &
echo "PID=$!  RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/train.log"
```

续训继续使用上一节的 `--resume` 命令；不传 `--precision` 时自动恢复
checkpoint 中的精度，旧 checkpoint 缺少该字段时视为 FP32。显式指定
与 checkpoint 不同的精度会报错，避免将两组实验混合。保存的是标准
FP32 权重，现有 `classifier_from_checkpoint` 可直接恢复和推理。

分析时比较验证 Macro-F1、损失曲线和稳定解冻阶段的 epoch 耗时；首次
运行、首次解冻的开销单独记录。短预跑吞吐不等于完整实验总时间，单个
种子的微小精度差异也不能直接认定为收益或退化。确认 BF16 后，再单独
考虑 batch 64 或分割定位裁剪，不在本轮同时改变这些条件。
### 8. 预测分割定位裁剪实验

下一轮实验用冻结 U-Net 预测宠物前景框，从原始分辨率图像裁剪主体，再用
ResNet-50 分类。裁剪框按前景概率阈值和边距生成；默认阈值 `0.5`、边距
`0.15`。图像与 trimap 使用同一个框，之后再进行 256×256 预处理和训练
增强。当前实验只生成 `trainval` cache，不使用测试集调参。

`build_mask_crop_cache.py` 的 `predicted` 模式模拟实际部署；`oracle` 模式
只用于诊断理想 mask 裁剪是否有帮助，不能作为系统成绩。训练脚本的
`--mode crop` 要求一个完整的 normalized crop cache，并把 cache 元数据
写入 `config.json`。预测框为空时记录并使用整图框，不能静默丢弃样本。

云端直接运行预测 mask 裁剪训练：

```bash
export DATA_ROOT=/mnt/workspace/petvision/data/oxford-iiit-pet
export SEGMENTATION_CHECKPOINT=/mnt/workspace/petvision-runs/run-RPdAyBz7/segmentation/best.pt
export RUN_DIR=/mnt/workspace/petvision-runs/resnet50-256-color-crop-bf16-$(date +%Y%m%d-%H%M%S)-$$
mkdir -p /mnt/workspace/petvision-runs
mkdir "$RUN_DIR"

export PRECISION=bf16 BATCH_SIZE=32 NUM_WORKERS=8 IMAGE_SIZE=256
export EPOCHS=30 HEAD_EPOCHS=4 PATIENCE=6
export BACKBONE_LR=0.00001 HEAD_LR=0.0001 WEIGHT_DECAY=0.01
export SPLIT_SEED=42 TRAINING_SEED=42
export CROP_THRESHOLD=0.5 CROP_MARGIN=0.15 CROP_BATCH_SIZE=64

nohup bash scripts/run_resnet50_crop_cloud.sh \
  >"$RUN_DIR/train.log" 2>&1 &
echo "PID=$!  RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/train.log"
```

脚本顺序为：检查 GPU → 生成 `crop-cache.json` → 训练 crop ResNet-50。
结果保存在 `crop/best.pt`、`crop/history.json` 和 `crop/config.json`；
不评价测试集。先比较该结果与 BF16 raw 基线；如果主体 crop 有收益，再
单独训练 global 分支并用验证集选择 global/local 概率融合权重。

如果要做 oracle 诊断，不使用云端主脚本，先生成真值框：

```bash
python -u scripts/build_mask_crop_cache.py \
  --root "$DATA_ROOT" \
  --output "$RUN_DIR/oracle-crop-cache.json" \
  --source oracle --split trainval \
  --image-size 256 --margin 0.15 \
  --batch-size 64 --num-workers 8 --device cuda
```

然后把训练命令中的 `--crop-cache` 指向该文件，并将输出改到新的目录。
oracle 结果只用于判断“分割定位是否有潜力”，不能替代 predicted crop。


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

## 水平翻转 TTA

不重新训练模型时，可以使用原图和水平翻转图的概率平均：

```powershell
.venv\Scripts\python.exe scripts\evaluate_fusion.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --output artifacts\fusion_tta `
  --batch-size 16 --num-workers 2 --lambda-step 0.05 `
  --seed 42 --tta-flip --device auto
```

`--tta-flip` 同时作用于 raw、guided 和 fusion；融合权重仍只在验证集
选择，结果中的 `metadata.tta_flip` 应为 `true`。

## 演示中的晚期融合

当前 ColorJitter 正式候选模型的验证集选择权重为
`raw=0.65`、`guided=0.35`。如果重新评价或更换模型，必须使用对应
`fusion/metrics.json` 中的 `metadata.raw_weight`：

```powershell
.venv\Scripts\python.exe scripts\app.py `
  --segmentation-checkpoint artifacts\segmentation\best.pt `
  --raw-checkpoint artifacts\classification\raw\best.pt `
  --guided-checkpoint artifacts\classification\guided\best.pt `
  --fusion-raw-weight 0.65 `
  --device auto
```

不要直接使用测试集调出的权重。
320×320 高分辨率对照可设置 `IMAGE_SIZE=320`，同时建议将
`BATCH_SIZE` 调整为 16；该实验应使用新的结果目录，不覆盖 256×256
ColorJitter 基线。
## 特征级融合

在冻结 raw/guided ResNet-18 backbone 的前提下，训练一个融合头：

```text
raw feature (512) + guided feature (512)
        -> Linear(1024, 256) -> ReLU -> Dropout -> 37 classes
```

云端脚本：

```bash
export DATA_ROOT=/mnt/workspace/petvision/data/oxford-iiit-pet
export SEGMENTATION_CHECKPOINT=/mnt/workspace/petvision-runs/run-RPdAyBz7/segmentation/best.pt
export RAW_CHECKPOINT=/mnt/workspace/petvision-runs/resnet18-color-.../raw/best.pt
export GUIDED_CHECKPOINT=/mnt/workspace/petvision-runs/resnet18-color-.../guided/best.pt
export RUN_DIR=/mnt/workspace/petvision-runs/feature-fusion-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
bash scripts/run_feature_fusion_cloud.sh
```

该实验的测试集评价保存在 `evaluation/metrics.json`。测试集不参与融合头
训练或超参数选择；当前第一阶段只训练融合头，不解冻两个 backbone。


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
