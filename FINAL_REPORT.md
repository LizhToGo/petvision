# PetVision 最终结果报告

## 1. 项目结论

项目实现了一个面向 Oxford-IIIT Pet 37 类品种的自然场景宠物图像理解系统：U-Net 预测前景，ResNet-50 完成品种分类，并提供原图分类、主体裁剪和可视化演示。

最终主模型采用 **ResNet-50 原图 FP32 分类器**。在冻结方案后进行一次官方测试集评价，结果为：

- Top-1 Accuracy：**93.540%**
- Macro-F1：**93.380%**
- Top-5 Accuracy：**99.782%**
- U-Net 前景 IoU：**85.607%**
- U-Net 前景 Dice：**92.246%**

余弦学习率和预测 mask 裁剪没有在测试集上超过原图 FP32 主模型，因此保留为消融对照，不作为最终分类路径。

## 2. 数据与评价口径

- 数据集：Oxford-IIIT Pet，37 个品种。
- 当前索引：trainval 3679 张，test 3669 张。
- 仓库镜像中缺少 mask 的 4 张图像已过滤；报告不把当前标签描述为完整三值 trimap。
- 所有训练方案在固定 train/validation 划分上选择 `best.pt`。
- 测试集只在训练方案冻结后评价一次，不用于阈值、边距、权重或模型选择。
- 分类主指标：Accuracy、Macro-F1、Top-5 Accuracy。
- 分割主指标：foreground IoU、foreground Dice、mIoU。

## 3. 最终模型配置

### 分类主模型

- 架构：ResNet-50
- 初始化：`torchvision ResNet50_Weights.IMAGENET1K_V2`
- 输入：256×256 RGB
- 增强：ColorJitter 与水平翻转
- 精度：FP32
- 优化器：AdamW，weight decay 0.01
- 学习率：backbone `1e-5`，classifier head `1e-4`
- 训练阶段：前 4 轮冻结 backbone，之后解冻
- 最大轮数：30
- 训练 seed：42
- 划分 seed：42
- 主权重：
  `C:/Users/27978/Downloads/111/resnet50-256-color-raw-20261103-212216-24717/raw/best.pt`

### 分割模型

- 模型：U-Net
- 输入分辨率：256×256
- 权重：`artifacts/formal_results/segmentation/best.pt`
- 预测裁剪阈值：0.5
- 裁剪边距：0.15

## 4. 验证集对照

| 方案 | 最佳轮次 | Accuracy | Macro-F1 | Top-5 Accuracy |
|---|---:|---:|---:|---:|
| ResNet-50 原图 FP32 固定学习率 | 28 | 97.096% | 97.121% | 99.819% |
| ResNet-50 原图 FP32 cosine | 28 | 97.096% | 97.127% | 99.819% |
| ResNet-50 原图 BF16 | 20 | 96.915% | 96.917% | 99.819% |
| ResNet-50 预测裁剪 BF16 | 18 | 96.915% | 96.932% | 99.819% |

cosine 相比固定学习率只提高 0.006 个百分点的 Macro-F1，Top-1 和 Top-5 不变；该差异不足以证明稳定收益。预测裁剪在验证集上也没有超过原图 FP32。

## 5. 官方测试集结果

测试评价文件：`artifacts/final_evaluation/metrics.json`。

| 方案 | Accuracy | Macro-F1 | Top-5 Accuracy | Loss |
|---|---:|---:|---:|---:|
| ResNet-50 原图 FP32 | **93.540%** | **93.380%** | **99.782%** | 0.2377 |
| ResNet-50 原图 cosine | 93.377% | 93.215% | 99.809% | 0.2205 |
| ResNet-50 预测裁剪 BF16 | 93.077% | 92.930% | 99.700% | 0.2293 |

U-Net 测试集指标：

| 指标 | 数值 |
|---|---:|
| Background IoU | 89.390% |
| Foreground IoU | 85.607% |
| mIoU | 87.499% |
| Background Dice | 94.398% |
| Foreground Dice | 92.246% |


主模型在 3669 张测试图像中预测错误 237 张。高频混淆主要集中在外观相近的品种：

- `american_pit_bull_terrier → staffordshire_bull_terrier`：22 张；
- `Ragdoll → Birman`：18 张；
- `american_pit_bull_terrier → american_bulldog`：15 张；
- `staffordshire_bull_terrier → american_bulldog`：13 张。

高置信度错误案例及对应的预测 mask 已导出到：

```text
artifacts/final_evaluation/errors/
```

## 6. 本地演示

推荐使用主体裁剪页面展示完整视觉流程，同时用最终 raw FP32 模型提供对照：

```powershell
$Results = "$HOME\Downloads\111"
.venv\Scripts\python.exe -m scripts.app_crop `
  --segmentation-checkpoint artifacts\formal_results\segmentation\best.pt `
  --crop-checkpoint "$Results\resnet50-256-color-crop-bf16-20261104-231401-2911\crop\best.pt" `
  --raw-checkpoint "$Results\resnet50-256-color-raw-20261103-212216-24717\raw\best.pt" `
  --device auto --port 7861 --open-browser
```

浏览器访问 `http://127.0.0.1:7861`。页面展示预测前景、裁剪框、主体裁剪图、raw FP32 与 crop 模型的 Top-5 结果。

该页面用于展示真实推理链路；分类性能结论以 `metrics.json` 的数据集评价为准，不以单张上传图片代替测试集指标。

## 7. 复现实验

生成测试集预测裁剪框：

```powershell
.venv\Scripts\python.exe scripts\build_mask_crop_cache.py `
  --root data\oxford-iiit-pet `
  --output artifacts\final_evaluation\test-predicted-crop-cache.json `
  --source predicted `
  --segmentation-checkpoint artifacts\formal_results\segmentation\best.pt `
  --split test --image-size 256 --threshold 0.5 --margin 0.15 `
  --batch-size 32 --num-workers 0 --device auto
```

运行最终评价：

```powershell
$Results = "$HOME\Downloads\111"
.venv\Scripts\python.exe scripts\evaluate_final.py `
  --root data\oxford-iiit-pet `
  --segmentation-checkpoint artifacts\formal_results\segmentation\best.pt `
  --raw-checkpoint "$Results\resnet50-256-color-raw-20261103-212216-24717\raw\best.pt" `
  --cosine-checkpoint "$Results\resnet50-256-color-raw-cosine-20261108-214153-1563\raw\best.pt" `
  --crop-checkpoint "$Results\resnet50-256-color-crop-bf16-20261104-231401-2911\crop\best.pt" `
  --crop-cache artifacts\final_evaluation\test-predicted-crop-cache.json `
  --output artifacts\final_evaluation `
  --batch-size 32 --num-workers 0 --device auto
```

导出高置信度错误：

```powershell
.venv\Scripts\python.exe scripts\export_final_errors.py `
  --root data\oxford-iiit-pet `
  --raw-checkpoint "$Results\resnet50-256-color-raw-20261103-212216-24717\raw\best.pt" `
  --segmentation-checkpoint artifacts\formal_results\segmentation\best.pt `
  --output artifacts\final_evaluation\errors `
  --count 12 --batch-size 32 --num-workers 0 --device auto
```

## 8. 局限与答辩表述

- 系统只覆盖训练时的 37 个已知品种。
- 系统假设图像中存在一个主要宠物目标。
- 没有无宠物、未知品种或多目标实例的拒识机制。
- Softmax 分数没有经过概率校准，不能直接解释为真实正确概率。
- 预测裁剪依赖 U-Net 的 mask；当前测试集缓存中有 1 张图像使用整图 fallback。
- cosine 和 crop 的验证集结果接近 raw FP32，但官方测试集没有超过 raw FP32；不能把它们描述为性能提升。
- 训练结果、数据、mask、checkpoint 和生成图片不提交到 Git 仓库；复现实验需要本地准备数据和权重。

## 9. 验证记录

- `python -m unittest discover -s tests -v`：7 个测试通过。
- `python -m compileall -q petvision scripts tests`：通过。
- 最终测试集评价：3669 张 test 图像，设备为 CUDA。
- 最终代码中的评价入口：`scripts/evaluate_final.py`。
- 最终错误案例入口：`scripts/export_final_errors.py`。
