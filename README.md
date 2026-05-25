# cnn-transformer-flickr8k

Image captioning on the Flickr8k dataset with a CNN image encoder and a Transformer decoder. Compares a custom CNN trained from scratch against a frozen pretrained ResNet18, and greedy decoding against beam search.

MSc Computer Vision coursework.

[English](#english) | [中文](#中文)

---

## English

### Overview

The implementation lives in a single Jupyter notebook (`CNN-Transformer.ipynb`) and covers the full pipeline end to end:

- Caption tokenisation with spaCy and a frequency-filtered vocabulary
- A `FlickrDataset` with row-based train/validation split (30,000 / remaining)
- A custom `PadCaptions` collate function for variable-length captions
- Two encoders: a lightweight 3-layer CNN trained from scratch, and a frozen pretrained ResNet18
- A `nn.TransformerDecoder` (4 layers, 8 heads) shared between both encoder experiments
- Greedy decoding and beam search decoding (with length penalty and no-repeat-bigram blocking)
- Evaluation with BLEU-1 to BLEU-4 (NLTK smoothing method4) and a custom ROUGE-L implementation
- Manual error analysis with six error categories: object, action, scene, attribute, counting, generic

### Architecture

**EncoderCNN (custom baseline)** — Three `Conv2d + ReLU + MaxPool` blocks (3→32→64→128 channels) followed by a fully connected layer and dropout. Input `(B, 3, 224, 224)` is mapped to a flat feature vector `(B, embed_size)`.

**ResNetEncoder** — Pretrained ResNet18 with the final classification layer removed. The 512-dimensional pooled feature is projected to `embed_size` through a linear layer with BatchNorm and dropout. The ResNet backbone is frozen; only the projection layer and decoder are trained.

**DecoderTransformer** — Standard `nn.TransformerDecoder`. The image feature is treated as the `memory` (reshaped to `(B, 1, embed_size)`), and the caption token embeddings are the target sequence with a causal mask. The output is projected to vocabulary logits.

**Training objective** — Cross-entropy between predicted logits `(B, seq_len-1, vocab_size)` and shifted targets `captions[:, 1:]`. `<PAD>` positions are ignored.

**Inference** — Greedy: feed `<SOS>`, repeatedly take the `argmax` of the last position. Beam search: maintain `beam_size` candidate sequences, expand and rescore each step using accumulated log probabilities normalised by `length ** length_penalty`, with optional bigram blocking.

### Dataset

[Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k): around 8,000 images with 5 captions each (~40,000 image-caption pairs).

The notebook downloads and extracts the dataset automatically at the top:

```python
!mkdir -p data/flickr8k/
!wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" -O "data/flickr8k/Flickr8k_Dataset.zip"
!wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" -O "data/flickr8k/Flickr8k_text.zip"
!unzip -q -o "data/flickr8k/Flickr8k_Dataset.zip" -d "data/flickr8k/"
!unzip -q -o "data/flickr8k/Flickr8k_text.zip" -d "data/flickr8k/"
```

These `!wget` / `!unzip` commands assume a Unix-like environment (Linux, macOS, WSL, Colab). On native Windows, install `wget` / `unzip` (e.g. via `winget` or `choco`), or download the two ZIP files manually and extract them so the layout matches:

```
data/flickr8k/
├── Flicker8k_Dataset/         # 8,091 .jpg images (original misspelling kept)
└── Flickr8k_text/
    └── Flickr8k.token.txt     # tab-separated image-caption annotations
```

The dataset is not redistributed in this repository.

### Setup

Tested with Python 3.11.

```bash
pip install torch torchvision spacy nltk pandas numpy matplotlib Pillow tqdm
python -m spacy download en_core_web_sm
```

### Running

Launch Jupyter from the project root so the relative `data/flickr8k/...` paths resolve correctly:

```bash
jupyter notebook CNN-Transformer.ipynb
```

Or run headlessly:

```bash
jupyter nbconvert --to notebook --execute CNN-Transformer.ipynb --output CNN-Transformer.ipynb
```

A CUDA GPU is recommended. On a single T4, one full run (both experiments × 5 epochs) takes around 25 minutes.

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding size | 64 |
| Batch size | 32 |
| Epochs | 5 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Transformer heads | 8 |
| Transformer layers | 4 |
| Vocabulary frequency threshold | 5 |
| Beam size | 3 |
| Beam length penalty | 0.7 |
| Evaluation samples | 500 |
| Seed | 6421 |

### Results

Numbers from one training run on the Flickr8k validation split (500 samples, 5 epochs):

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Custom CNN + Greedy | 0.058 | 0.002 | 0.000 | 0.000 | 0.091 |
| ResNet18 + Greedy | 0.057 | 0.002 | 0.000 | 0.000 | 0.091 |
| ResNet18 + Beam Search | 0.347 | 0.172 | 0.098 | 0.059 | 0.336 |

Beam search produces a large jump over greedy decoding on this small model. Qualitative inspection shows the model captures broad concepts (people, dogs, water, grass) but often falls back to common caption templates such as "a dog running through the grass" and misses fine-grained attributes, actions, and counts.

On a manually labelled subset of 10 examples, the most common error categories were object errors (7), action errors (5), scene errors (5), and attribute errors (3).

### Repository contents

- `CNN-Transformer.ipynb` — Main notebook covering the full pipeline.
- `_annotate.py` — Helper script that rewrites code cells in `CNN-Transformer.ipynb` with bilingual (English / Chinese) inline comments. Used as a build-time tool when refreshing the annotations; not required to run the notebook.
- `README.md` — This file.

### Limitations

The encoder compresses the entire image into a single global feature vector, which discards spatial structure. This makes the decoder rely heavily on caption priors and explains the model's tendency to produce generic, template-like captions.

Possible directions for improvement: fine-tuning the last ResNet block, using spatial feature maps with visual attention, lowering the vocabulary frequency threshold to reduce `<UNK>` tokens, and training on a larger dataset such as MS-COCO.

---

## 中文

### 项目简介

整套实现都在一个 Jupyter notebook（`CNN-Transformer.ipynb`）里，端到端覆盖以下流程：

- 使用 spaCy 做 caption 分词，构建带词频阈值的词表
- `FlickrDataset` 按行做训练 / 验证集划分（前 30,000 / 剩余）
- 自定义 `PadCaptions` collate 函数处理变长 caption
- 两种 encoder：从零训练的 3 层轻量 CNN，以及冻结的预训练 ResNet18
- 两个实验共用同一个 `nn.TransformerDecoder`（4 层，8 头）
- 贪心解码（greedy）与束搜索（beam search，含长度惩罚和重复 bigram 抑制）
- 用 BLEU-1 ~ BLEU-4（NLTK `SmoothingFunction.method4`）和自实现的 ROUGE-L 做定量评估
- 人工错误分析，分为 6 类：object / action / scene / attribute / counting / generic

### 模型结构

**EncoderCNN（自定义基线）** —— 三组 `Conv2d + ReLU + MaxPool`（通道数 3→32→64→128），之后接全连接层 + dropout。输入 `(B, 3, 224, 224)`，输出 `(B, embed_size)` 的扁平特征向量。

**ResNetEncoder** —— 用预训练 ResNet18，去掉最后的分类层。把 512 维的池化特征经过一个线性层 + BatchNorm + dropout 投影到 `embed_size`。ResNet 主干被冻结，只训练投影层和 decoder。

**DecoderTransformer** —— 标准 `nn.TransformerDecoder`。图像特征作为 `memory`（reshape 成 `(B, 1, embed_size)`），caption token embedding 作为 target 序列，配合 causal mask；最后线性层投影到词表 logits。

**训练目标** —— 预测 logits `(B, seq_len-1, vocab_size)` 与右移后的 target `captions[:, 1:]` 之间的交叉熵损失，`<PAD>` 位置不参与计算。

**推理** —— 贪心：以 `<SOS>` 起始，每步取最后位置的 `argmax`。束搜索：维护 `beam_size` 条候选序列，每步扩展并用 `score / length ** length_penalty` 重新排序，可选地阻止重复 bigram。

### 数据集

[Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)：约 8,000 张图，每张配 5 条 caption，共约 40,000 条 image-caption pair。

Notebook 顶部会自动下载并解压数据集：

```python
!mkdir -p data/flickr8k/
!wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" -O "data/flickr8k/Flickr8k_Dataset.zip"
!wget "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" -O "data/flickr8k/Flickr8k_text.zip"
!unzip -q -o "data/flickr8k/Flickr8k_Dataset.zip" -d "data/flickr8k/"
!unzip -q -o "data/flickr8k/Flickr8k_text.zip" -d "data/flickr8k/"
```

这些 `!wget` / `!unzip` 命令需要类 Unix 环境（Linux / macOS / WSL / Colab）。Windows 原生环境下请先装 `wget` / `unzip`（用 `winget` 或 `choco`），或者手动下载两个 ZIP 文件并解压成下面这个目录结构：

```
data/flickr8k/
├── Flicker8k_Dataset/         # 8,091 张 .jpg 图（拼写错误来自原数据集，保留）
└── Flickr8k_text/
    └── Flickr8k.token.txt     # 制表符分隔的 image-caption 标注
```

本仓库不附带数据集。

### 环境准备

测试环境：Python 3.11。

```bash
pip install torch torchvision spacy nltk pandas numpy matplotlib Pillow tqdm
python -m spacy download en_core_web_sm
```

### 运行

在项目根目录启动 Jupyter，这样 notebook 里的 `data/flickr8k/...` 相对路径才能解析正确：

```bash
jupyter notebook CNN-Transformer.ipynb
```

或者无人值守地执行：

```bash
jupyter nbconvert --to notebook --execute CNN-Transformer.ipynb --output CNN-Transformer.ipynb
```

建议用 CUDA GPU。单卡 T4 跑完两个实验（各 5 epoch）大约 25 分钟。

### 超参数

| 参数 | 取值 |
|---|---|
| Embedding 维度 | 64 |
| Batch size | 32 |
| Epoch 数 | 5 |
| 学习率 | 1e-4 |
| 优化器 | Adam |
| Transformer 头数 | 8 |
| Transformer 层数 | 4 |
| 词表频次阈值 | 5 |
| Beam size | 3 |
| Beam 长度惩罚 | 0.7 |
| 评估样本数 | 500 |
| 随机种子 | 6421 |

### 结果

一次完整训练后在 Flickr8k 验证集（500 个样本，5 epoch）上的指标：

| 模型 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Custom CNN + Greedy | 0.058 | 0.002 | 0.000 | 0.000 | 0.091 |
| ResNet18 + Greedy | 0.057 | 0.002 | 0.000 | 0.000 | 0.091 |
| ResNet18 + Beam Search | 0.347 | 0.172 | 0.098 | 0.059 | 0.336 |

束搜索相对贪心解码在这套小模型上有明显提升。定性观察上，模型能抓住大类概念（人、狗、水、草），但经常退化成 "a dog running through the grass" 这样的模板句，无法刻画具体属性、动作和数量。

在人工标注的 10 条样本里，最常见的错误类别是 object error (7)、action error (5)、scene error (5)、attribute error (3)。

### 仓库结构

- `CNN-Transformer.ipynb` —— 主 notebook，包含完整流程。
- `_annotate.py` —— 辅助脚本，会按 cell id 重写 `CNN-Transformer.ipynb` 的代码 cell，在其中加入中英双语注释。属于构建期工具，用来刷新双语注释；运行 notebook 本身不需要它。
- `README.md` —— 当前文档。

### 局限性

整张图被压成单个全局特征向量，丢掉了空间结构。这让 decoder 高度依赖 caption 先验，也解释了为什么生成的 caption 经常走模板路线。

潜在改进方向：解冻 ResNet 最后一个 block 做微调；用空间特征图加视觉 attention；降低词表频次阈值以减少 `<UNK>`；换成 MS-COCO 这种更大规模数据集训练。
