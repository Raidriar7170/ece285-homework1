# ECE 285 Homework 1 — Chest X-ray VAE

代码仓库：https://github.com/Raidriar7170/ece285-homework1

作业平台仅接收 PDF：提交 `submission/Homework1_report.pdf`，报告内已包含可点击的 GitHub 代码链接。

## 作业对应关系

- `model.py`：自行实现的 PyTorch 卷积 VAE（64×64 灰度，64 维 latent）。
- `prepare_data.py`：下载题目指定 Kaggle 数据，排除压缩包内重复副本与 macOS 元数据。
- `train.py`：CPU 默认训练，BCE + KL，训练／验证曲线、最佳模型、生成样例与重建图。
- `evaluate.py`：独立测试集评估，使用 PyTorch-Ignite 计算 FID、Inception Score。
- `report/paper/main.tex`：官方 NeurIPS 2026 格式报告源文件。

## 环境与复现

Python 3.12；PyTorch 2.6.0、torchvision 0.21.0、PyTorch-Ignite 0.5.2。
以下命令在 `homework1/` 中执行。数据压缩包约 2.35 GiB；解压仅保留一份图片。
训练输出目录必须不存在，以避免覆盖实验记录。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_data.py
python train.py --data data/chest_xray --out runs/reproduction --epochs 30 --device cpu
python evaluate.py --data data/chest_xray --run runs/reproduction
```

默认训练参数：Adam，lr=0.001，batch=64，30 epochs，latent=64，seed=285，CPU 6 线程。
仅使用原始 train 划分学习参数，用原始 val 的随机负 ELBO（固定验证噪声）选择 checkpoint。
保留验证集只有 16 张的事实，不把它描述为稳健的超参数搜索。无增强、无标签条件、无预训练 VAE。
训练损失是每张图像像素求和 BCE 加 latent 求和 KL，再对 batch 取平均（beta=1）。
BCE 对连续灰度采用常见的 soft-target surrogate，不将其当作严格的连续密度模型。

训练生成的 `best.pt` 和 `last.pt` 包含模型、优化器和配置（仓库仅提供 `best.pt`）；当前脚本支持从头重现，**不提供精确断点续训**。
请只加载本项目生成且可信的 checkpoint（含 PyTorch/Python 对象）。

## 评估约定

- 真图为原始 test 划分全部图片；每张真图对应一个独立标准正态 latent 生成样本。
- 独立生成 seed=1285，测试结果不参与训练和 checkpoint 选择。
- 真／假图同样由 64×64 灰度 [0,1] 转 uint8 RGB，PIL bilinear 缩放至 299×299，再做 ImageNet 标准化。
- FID 使用 torchvision ImageNet Inception-v3 的 **2048 维 pooled features** 和 Ignite FID 累积。
  教程的 Ignite 默认值使用 1000 维分类概率；这里显式指定特征层，并非直接报告默认值。
  本分数不是 TensorFlow FID 或 Clean-FID，不能与不同实现直接横比。
- IS 使用同一网络的 1000 类 softmax，整个生成集计算一个全局分数，不报告未测量的 split 标准差。
- ImageNet 指标只作题目要求的参考，不能证明诊断质量；小样本 FID 有偏。
- SHA-256 检查可排除跨划分完全相同的文件，但不能证明患者级独立或排除近重复图像。
- 生成图展示固定 seed 的首 64 张，不按视觉好坏筛选。重建图是验证图的 posterior mean 重建，与无条件生成明确区分。

## 输出与提交

`runs/vae_cpu/` 保存当前实际实验：配置、划分文件哈希、逐轮损失、最佳与末轮 checkpoint、曲线、图像、指标。
已完成本机 CPU 训练与评估，报告已渲染检查为 **2 页**。

- `submission/Homework1_report.pdf`：英文 NeurIPS 格式报告。
- GitHub 仓库提供源代码、报告 LaTeX 源、最佳权重与实验记录；不含数据集、末轮权重及原图重建对照图。无需另交 ZIP。
- 30 轮训练约 6.3 分钟，验证集选择第 26 轮。
- 624 张测试图 / 624 张独立生成图：FID-2048 = 196.5986841590；IS = 1.7685023504。
- 4 项测试通过，独立只读复核确认报告数值、模型与源码哈希一致。
- 图像仍明显模糊；这些分数不是医学诊断质量证明。

复现生成图与报告（需安装 Tectonic，或在 LaTeX 编辑器编译 `report/paper/main.tex`）：

```bash
python generate.py --checkpoint runs/reproduction/best.pt --output runs/reproduction/samples16.png --count 16 --nrow 4
python build_report.py --run runs/reproduction --observation 'Describe the newly generated samples after inspecting them.'
```

报告构建脚本会读取该次实验的真实记录；仅适用于正文所写的 30 轮与 624 张测试图设置。
另一次训练的数值可能受硬件与依赖版本影响。`environment.json` 记录本次实际版本。
**不会自动提交 Gradescope。**

## 来源

- 作业：用户提供的 `Homework_1` PDF（要求两页 NeurIPS 报告和代码）。
- 数据：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- 指标题目指定教程：https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- NeurIPS 官方模板：https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
- D. P. Kingma and M. Welling, Auto-Encoding Variational Bayes, ICLR 2014, https://arxiv.org/abs/1312.6114
