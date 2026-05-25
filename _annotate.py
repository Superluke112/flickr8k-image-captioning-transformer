# -*- coding: utf-8 -*-
"""Rewrite code cell sources in CNN-Transformer.ipynb with bilingual comments."""
import json
from pathlib import Path

NB_PATH = Path(r"E:/Msc/CV/Project/CNN-Transformer/CNN-Transformer.ipynb")

# Map: cell_id -> new source (as a single string; will be split into lines)
NEW_SOURCES = {}

NEW_SOURCES["479a508d-b0b3-4a0e-a97b-1a7f39fcea9f"] = r'''# Standard library / 标准库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image                          # Image I/O / 图像读取
from collections import Counter                # Word frequency counting / 词频统计

# PyTorch core / PyTorch 核心
import torch
import torch.nn as nn                          # Neural network layers / 神经网络模块
import torch.nn.functional as F                # Functional API (relu, softmax...) / 函数式接口
import torch.optim as optim                    # Optimizers / 优化器
from torch.utils.data import Dataset, DataLoader              # Dataset abstractions / 数据集与加载器
from torch.nn.utils.rnn import pad_sequence    # Pad variable-length sequences / 变长序列填充

import torchvision.transforms as T             # Image preprocessing / 图像预处理
import spacy                                   # English tokenizer / 英文分词
from tqdm import tqdm                          # Progress bar / 进度条
import random
import time'''

NEW_SOURCES["d5ff491f-b89a-4e24-9827-5b024e2596c6"] = r'''data_location =  "data/flickr8k"     # Root of the Flickr8k dataset / Flickr8k 数据集根目录

# Path to the caption file (filename<TAB>caption) / caption 文本文件路径
caption_file = data_location + '/Flickr8k_text/Flickr8k.token.txt'
# Read TSV with no header; column 0 = "filename#k", column 1 = caption text
# 用制表符分隔读取，无表头；第 0 列是 "文件名#编号"，第 1 列是描述文本
df = pd.read_csv(caption_file, delimiter='\t', header=None)

# Strip the "#k" suffix so that column 0 becomes the pure filename
# 去掉 "#k" 后缀，使第 0 列只保留纯文件名
df.iloc[:,0] = df[0].apply(lambda x: x.split('#')[0])
# Drop dirty rows whose filename contains ".jpg.1" / 过滤掉文件名含 ".jpg.1" 的异常行
df = df[~df[0].apply(lambda x: '.jpg.1' in x)]

print("There are {} unique captions".format(len(df)))
'''

NEW_SOURCES["03d04903-8fbc-4f60-a39e-52bb5a0783f2"] = r'''# Load the small English spaCy model used for tokenization
# 加载 spaCy 的小型英文模型，用于把句子切成 token
spacy_eng = spacy.load("en_core_web_sm")'''

NEW_SOURCES["f9cf7a06-369a-4e49-a9b6-69c43bbef128"] = r'''class Vocabulary:
    def __init__(self, freq_threshold):
        # Reserved special tokens with fixed indices / 固定索引的特殊 token
        # 0=<PAD> padding, 1=<SOS> start, 2=<EOS> end, 3=<UNK> unknown
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}

        # Reverse mapping word -> index / 反向映射：单词到索引
        self.stoi = {v:k for k,v in self.itos.items()}

        # A word must appear at least this many times to enter the vocab
        # 单词出现次数达到该阈值才会被收录进词表
        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    @staticmethod
    def tokenize(text):
        # spaCy tokenization + lowercasing / 用 spaCy 分词并转小写
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()      # Word frequency counter / 词频计数器
        idx = 4                      # First free index after the 4 special tokens / 特殊 token 之后从 4 开始编号

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # Add the word the first time it crosses the threshold
                # 当词频刚好达到阈值时（只触发一次）把它加入词表
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        '''Convert a caption string into a list of token indices.
        把一句 caption 转换成 token 索引列表。"""
        tokenized_text = self.tokenize(text)
        # Unknown words fall back to <UNK> / 未登录词用 <UNK> 替代
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]"""

NEW_SOURCES["079db71a-6f7a-456a-9daa-4be90722ce65"] = r'''class FlickrDataset(Dataset):
    '''
    Custom Dataset for Flickr8k. Returns (image_tensor, caption_indices).
    自定义 Flickr8k 数据集，返回 (图像张量, caption 索引序列)。
    """
    def __init__(self, root_dir, captions_file, transform=None, train=True, freq_threshold=5):
        self.root_dir = root_dir
        # Load (filename, caption) pairs / 读取 (文件名, caption) 表
        self.df = pd.read_csv(captions_file, delimiter='\t', header=None)

        # Shuffle deterministically so train/val split is reproducible
        # 用固定随机种子打乱，保证 train/val 划分可复现
        np.random.seed(42)
        self.df = self.df.sample(frac=1)

        self.df = self.df.reset_index(drop=True)
        # Strip "#k" from filename / 去掉 "#k" 后缀
        self.df.iloc[:,0] = self.df[0].apply(lambda x: x.split('#')[0])
        # Filter out malformed filenames / 过滤异常行
        self.df = self.df[~self.df[0].apply(lambda x: '.jpg.1' in x)]
        self.df = self.df.reset_index()
        self.transform = transform

        # Convenience handles for the two columns / 取出图像名列和 caption 列
        self.imgs = self.df[0]
        self.captions = self.df[1]

        # Build vocabulary from ALL captions (so train/val share the same vocab)
        # 用全部 caption 构建词表（train/val 共享同一个词表）
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

        # Train/val split: first 30000 vs the rest / 训练集前 30000 条，其余作验证
        if train:
            self.df = self.df.iloc[:30000,:]
        else:
            self.df = self.df.iloc[30000:,:]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        # Open image and force 3-channel RGB / 读取图片并强制转 RGB
        img = Image.open(img_location).convert("RGB")

        # Apply transforms (Resize/Normalize/ToTensor) / 执行预处理
        if self.transform is not None:
            img = self.transform(img)

        # Wrap caption with <SOS> ... <EOS> / 给 caption 加首尾标记
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)"""

NEW_SOURCES["8ec30096-1899-45b0-a3bb-62061482990d"] = r'''# Image preprocessing pipeline / 图像预处理流程
transforms = T.Compose([
    T.Resize((224,224)),                        # Resize to 224x224 (ResNet input size) / 缩放到 224x224
    T.ToTensor(),                               # PIL -> Tensor and scale to [0,1] / 转张量并归一化到 [0,1]
    # Channel-wise normalization (mean/std manually computed for this dataset)
    # 按通道做标准化（均值/方差是针对本数据集预先计算的）
    T.Normalize((0.485, 0.4461, 0.4039), (0.2695, 0.2623, 0.2772)),
])'''

NEW_SOURCES["82e3a41a-0796-45b4-aeb0-2d1808e7e4ab"] = r'''# Build the training split (first 30000 samples) / 构造训练集
train_dataset =  FlickrDataset(
    root_dir = data_location+"/Flicker8k_Dataset",
    captions_file = data_location+"/Flickr8k_text/Flickr8k.token.txt",
    transform=transforms
)

# Build the validation split (remaining samples) / 构造验证集
val_dataset =  FlickrDataset(
    root_dir = data_location+"/Flicker8k_Dataset",
    captions_file = data_location+"/Flickr8k_text/Flickr8k.token.txt",
    train=False,
    transform=transforms
)'''

NEW_SOURCES["f74b7469-f57b-4f11-928b-ce23928e1f2e"] = r'''class PadCaptions:
    '''Collate function: pad captions in a batch to the same length.
    自定义 collate_fn：把同一 batch 内变长 caption 填充到一致长度。"""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx                   # Index of <PAD> / <PAD> 的索引

    def __call__(self, batch):
        # Stack images into shape (B, C, H, W) / 把图片堆叠成 (B, C, H, W)
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        # Pad caption tensors along the seq dim / 在序列维度上做填充
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets

# Index used for padding / 填充索引
pad_idx = train_dataset.vocab.stoi["<PAD>"]

# Training DataLoader: shuffled, batched, padded / 训练用 DataLoader：打乱+成批+填充
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn = PadCaptions(pad_idx=pad_idx)
)

# Validation DataLoader: not shuffled / 验证用 DataLoader：保持顺序
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    collate_fn = PadCaptions(pad_idx=pad_idx)
)"""

NEW_SOURCES["67f4f10a-910e-499a-a7da-fc915f77a95b"] = r'''class EncoderCNN(nn.Module):
    '''Lightweight 3-layer CNN encoder. 自定义 3 层卷积编码器。"""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Block 1: (3,224,224) -> (32,112,112) / 第 1 块卷积 + 池化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: -> (64,56,56) / 第 2 块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: -> (128,28,28) / 第 3 块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Project flattened features to embed_size / 全连接映射到嵌入维度
        self.fc = nn.Linear(128 * 28 * 28, embed_size)
        self.dropout = nn.Dropout(0.5)           # Regularization / 防过拟合

    def forward(self, images):
        # conv -> relu -> pool, three times / 三次「卷积-激活-池化」
        x = self.pool1(F.relu(self.conv1(images)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten to a vector per image / 把每张图压成向量
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        return x

class DecoderTransformer(nn.Module):
    """Transformer decoder that generates captions from image features.
    用 Transformer Decoder 根据图像特征生成 caption。"""
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=4, dropout=0.1):
        super(DecoderTransformer, self).__init__()
        # Word embedding table / 词嵌入查找表
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Single decoder layer (multi-head self-attn + cross-attn + FFN)
        # 单层解码器：多头自注意力 + 编码器-解码器交叉注意力 + 前馈网络
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True                     # Input shape (B, L, D) / 输入维度顺序
        )
        # Stack `num_layers` layers / 堆叠 num_layers 层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final projection to vocabulary logits / 输出层：映射到词表 logits
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        """
        features: (batch, embed_size)  encoder output / 编码器输出
        captions: (batch, seq_len)     teacher forcing input / 教师强制输入
        """
        # Teacher forcing: drop the last token; we predict it from earlier ones
        # 教师强制：去掉最后一个 token；用前面的预测它
        tgt = self.embedding(captions[:, :-1])

        # Treat the image feature as the encoder "memory"; add a seq dim
        # 把图像特征视为编码器输出，需要添加一个序列维 -> (B, 1, D)
        memory = features.unsqueeze(1)

        # Causal mask: position i can only attend to <= i (no future peeking)
        # 因果掩码：位置 i 只能看到 <= i，防止解码器作弊
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # Decode: tgt is the (shifted) caption, memory is the image feature
        # 执行解码：tgt 是错位的 caption，memory 是图像特征
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

        # Project hidden states to vocab logits / 隐状态 -> 词表 logits
        return self.linear(output)"""

NEW_SOURCES["a9c895b2-10dd-42b8-90bd-d48771fc2df0"] = r'''def generate_caption(encoder, decoder, image, vocab, max_length=20):
    '''Greedy autoregressive decoding for a single image.
    单张图片的贪心自回归解码。"""
    encoder.eval()                               # Disable dropout/BN updates / 关闭 dropout 等
    decoder.eval()

    result_caption = []                          # Generated word tokens / 输出的词序列

    with torch.no_grad():                        # No gradient during inference / 推理无需梯度
        # 1) Encode the image / 1) 图像编码
        # (3,224,224) -> (1,3,224,224) add batch dim / 加 batch 维
        x = image.unsqueeze(0).to(DEVICE)
        features = encoder(x)                    # (1, embed_size)
        # Treat image feature as memory / 把图像特征作为 memory
        memory = features.unsqueeze(1)           # (1, 1, embed_size)

        # 2) Initialize sequence with <SOS> / 2) 用 <SOS> 作为起始
        sos_idx = vocab.stoi["<SOS>"]
        eos_idx = vocab.stoi["<EOS>"]

        # Running sequence of generated indices / 已生成的索引序列
        current_indices = torch.tensor([[sos_idx]], device=DEVICE)  # (1, 1)

        for _ in range(max_length):
            # Embed the sequence so far / 对当前已生成序列做嵌入
            tgt = decoder.embedding(current_indices)               # (1, L, D)

            # Causal mask for the current length / 当前长度对应的因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_indices.size(1)).to(DEVICE)

            # Forward through the decoder / 解码器前向
            output = decoder.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)

            # Use the LAST position to predict the next word / 取最后位置预测下一词
            logits = decoder.linear(output[:, -1, :])              # (1, vocab_size)
            predicted = logits.argmax(1)                           # Greedy pick / 贪心选最大

            # Append the predicted token for the next step / 追加到序列里
            current_indices = torch.cat((current_indices, predicted.unsqueeze(0)), dim=1)

            token = vocab.itos[predicted.item()]
            if token == "<EOS>":                # Stop when <EOS> generated / 遇到 <EOS> 提前停止
                break

            result_caption.append(token)

    return ' '.join(result_caption)"""

NEW_SOURCES["91db165b-2c05-4c04-be6b-d31d0b4483e4"] = r'''from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np

def clean_tokens(tokens):
    '''Strip special tokens from a token list. 移除特殊 token。"""
    special_tokens = {"<SOS>", "<EOS>", "<PAD>"}
    return [token for token in tokens if token not in special_tokens]


def caption_tensor_to_tokens(caption_tensor, vocab):
    """Convert an index tensor back to a list of words.
    把索引张量还原为单词列表。"""
    tokens = []

    for idx in caption_tensor.tolist():
        word = vocab.itos[idx]
        if word == "<EOS>":                    # Stop at end-of-sentence / 遇到 <EOS> 停止
            break
        if word not in ["<SOS>", "<PAD>"]:     # Skip <SOS>/<PAD> / 跳过 <SOS>/<PAD>
            tokens.append(word)

    return tokens


def lcs_length(x, y):
    """Dynamic-programming longest common subsequence length (for ROUGE-L).
    动态规划求最长公共子序列长度（用于 ROUGE-L）。"""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]   # DP table / DP 表

    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1          # Match -> extend LCS / 匹配则延长
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])   # Else carry forward / 否则继承较大值

    return dp[m][n]


def rouge_l_score(reference, candidate):
    """Compute ROUGE-L F1 between two token lists.
    计算单对 (参考, 候选) 的 ROUGE-L F1 分数。"""
    if len(reference) == 0 or len(candidate) == 0:
        return 0.0

    lcs = lcs_length(reference, candidate)

    precision = lcs / len(candidate)             # Precision: LCS / |candidate|
    recall = lcs / len(reference)                # Recall:    LCS / |reference|

    if precision + recall == 0:
        return 0.0

    # F1 score / F1 分数
    return 2 * precision * recall / (precision + recall)"""

NEW_SOURCES["7433c860-bf11-44c1-ac35-d48eefd8497a"] = r'''def evaluate_caption_metrics(
    encoder,
    decoder,
    val_dataset,
    vocab,
    max_samples=500,
    decoding_method="greedy",
    beam_size=3,
    length_penalty=0.7
):
    '''Run BLEU-1..4 and ROUGE-L on the validation set.
    在验证集上计算 BLEU-1~4 和 ROUGE-L。
    decoding_method: 'greedy' or 'beam' / 解码方式：贪心或束搜索"""
    encoder.eval()
    decoder.eval()

    references = []         # List of [ref_tokens] for corpus BLEU / corpus BLEU 用的参考列表
    candidates = []         # List of candidate token lists / 候选 caption 列表
    rouge_l_scores = []     # Per-sample ROUGE-L scores / 每条样本的 ROUGE-L

    # NLTK smoothing for short candidates / NLTK 平滑函数，处理短句问题
    smoothie = SmoothingFunction().method4

    num_samples = min(max_samples, len(val_dataset))

    for i in range(num_samples):
        image, caption = val_dataset[i]

        # Greedy decoding branch / 贪心解码分支
        if decoding_method == "greedy":
            predicted_caption = generate_caption(
                encoder=encoder,
                decoder=decoder,
                image=image,
                vocab=vocab,
                max_length=20
            )

        # Beam search branch / 束搜索分支
        elif decoding_method == "beam":
            predicted_caption = generate_caption_beam_search(
                encoder=encoder,
                decoder=decoder,
                image=image,
                vocab=vocab,
                beam_size=beam_size,
                max_length=20,
                length_penalty=length_penalty
            )

        else:
            raise ValueError("decoding_method must be 'greedy' or 'beam'.")

        # Tokenize candidate and reference / 把预测和参考都处理成 token 列表
        candidate_tokens = clean_tokens(predicted_caption)
        reference_tokens = caption_tensor_to_tokens(caption, vocab)

        # Skip empty pairs / 跳过空样本
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            continue

        references.append([reference_tokens])    # corpus_bleu expects list-of-list
        candidates.append(candidate_tokens)

        rouge_l_scores.append(
            rouge_l_score(reference_tokens, candidate_tokens)
        )

    # BLEU-n with uniform n-gram weights / 不同 n 的 BLEU（权重均匀）
    bleu_1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0),       smoothing_function=smoothie)
    bleu_2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0),     smoothing_function=smoothie)
    bleu_3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0),   smoothing_function=smoothie)
    bleu_4 = corpus_bleu(references, candidates, weights=(0.25,0.25,0.25,0.25),smoothing_function=smoothie)

    rouge_l = np.mean(rouge_l_scores)            # Average ROUGE-L / ROUGE-L 取均值

    results = {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4,
        "ROUGE-L": rouge_l
    }

    return results"""

NEW_SOURCES["7891ac7b"] = r'''def generate_caption_beam_search(
    encoder,
    decoder,
    image,
    vocab,
    beam_size=3,
    max_length=20,
    length_penalty=0.7,
    no_repeat_ngram_size=2
):
    '''Beam search decoding.
    束搜索解码：每步保留 beam_size 个最优部分序列。
    Note: decoder.forward uses captions[:, :-1] internally,
    so we append a dummy PAD at each step to feed the full current sequence.
    注意：decoder.forward 内部会丢弃最后一个 token，
    因此每步在序列末尾追加一个 PAD 占位，让真实序列全部参与解码。"""
    encoder.eval()
    decoder.eval()

    sos_idx = vocab.stoi["<SOS>"]                # Start symbol / 起始符
    eos_idx = vocab.stoi["<EOS>"]                # End symbol   / 结束符
    pad_idx = vocab.stoi["<PAD>"]                # Padding / 填充符

    with torch.no_grad():
        x = image.unsqueeze(0).to(DEVICE)
        features = encoder(x)                    # Encode once, reuse for all beams / 只编码一次

        # Each beam = (token_sequence, cumulative_log_prob)
        # 每条束 = (已生成序列, 累计 log 概率)
        beams = [([sos_idx], 0.0)]
        completed_beams = []                     # Beams that have generated <EOS> / 已完成的束

        for _ in range(max_length):
            new_beams = []

            for seq, score in beams:
                if seq[-1] == eos_idx:           # This beam already ended / 该束已结束
                    completed_beams.append((seq, score))
                    continue

                # Dummy PAD trick: decoder.forward will drop the last token
                # 在末尾加一个 PAD，使解码器的 [:, :-1] 切片刚好保留完整序列
                decoder_seq = seq + [pad_idx]

                caption_tensor = torch.tensor(
                    decoder_seq,
                    dtype=torch.long
                ).unsqueeze(0).to(DEVICE)

                outputs = decoder(features, caption_tensor)

                # Take logits from the last position / 取最后位置的 logits
                last_logits = outputs[:, -1, :]

                # Convert to log-probs for numerically stable summation
                # 转 log-softmax 便于直接累加
                log_probs = F.log_softmax(last_logits, dim=-1)

                # Suppress generating <SOS>/<PAD> mid-sequence / 屏蔽特殊 token
                log_probs[:, sos_idx] = -float("inf")
                log_probs[:, pad_idx] = -float("inf")

                # Top-k candidates for this beam / 当前束的 top-k 扩展
                top_log_probs, top_indices = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )

                for i in range(beam_size):
                    next_word_idx = top_indices[0, i].item()
                    next_word_log_prob = top_log_probs[0, i].item()

                    new_seq = seq + [next_word_idx]

                    # No-repeat n-gram constraint / 禁止重复的 n-gram，避免循环输出
                    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
                        words = new_seq
                        if len(words) >= 2 * no_repeat_ngram_size:
                            last_ngram = tuple(words[-no_repeat_ngram_size:])
                            previous_ngrams = [
                                tuple(words[j:j + no_repeat_ngram_size])
                                for j in range(len(words) - no_repeat_ngram_size)
                            ]

                            if last_ngram in previous_ngrams:
                                continue          # Skip this expansion / 丢弃该扩展

                    new_score = score + next_word_log_prob
                    new_beams.append((new_seq, new_score))

            if len(new_beams) == 0:               # No expansions left / 没有可扩展的束
                break

            # Length-normalized score so shorter beams aren't unfairly favored
            # 用长度惩罚归一化得分，避免短句天然占优
            def normalised_score(beam):
                seq, score = beam
                return score / (len(seq) ** length_penalty)

            new_beams = sorted(
                new_beams,
                key=normalised_score,
                reverse=True
            )

            # Keep the top-beam_size beams / 只保留前 beam_size 条
            beams = new_beams[:beam_size]

        # Treat unfinished beams as candidates too / 未结束的束也加入候选
        completed_beams.extend(beams)

        def final_score(beam):
            seq, score = beam
            return score / (len(seq) ** length_penalty)

        # Pick the best beam overall / 选出最终得分最高的束
        best_seq, best_score = max(completed_beams, key=final_score)

    # Convert indices back to words, skip specials / 索引转单词，过滤特殊 token
    caption = []

    for idx in best_seq:
        word = vocab.itos[idx]

        if word in ["<SOS>", "<PAD>"]:
            continue

        if word == "<EOS>":
            break

        caption.append(word)

    return caption"""

NEW_SOURCES["2e8024df-f306-4699-a21e-86c783e1072c"] = r'''# ============ Training script (custom CNN + Transformer) ============
# ============ 训练脚本（自定义 CNN + Transformer） ============

IMAGE_SIZE = 224
BATCH_SIZE = 32

embed_size = 64                                  # Token/feature dim / 嵌入维度
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available / 有 GPU 就用 GPU

vocab = train_dataset.vocab
VOCAB_SIZE = len(vocab)
PAD_IDX = vocab.stoi["<PAD>"]

res_e2_dmodel = []                               # Will hold the summary row / 结果记录
seed = 6421                                      # Fixed seed for reproducibility / 固定随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Instantiate models on DEVICE / 建模并搬到设备上
encoder = EncoderCNN(embed_size).to(DEVICE)
decoder = DecoderTransformer(embed_size, VOCAB_SIZE).to(DEVICE)

# Loss ignores <PAD> positions; optimizer = Adam over both nets
# 损失忽略 <PAD> 位置；优化器同时优化 encoder/decoder 参数
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

start_time = time.time()                         # Timing the run / 计时


train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # --- TRAINING PHASE --- / --- 训练阶段 ---
    encoder.train()
    decoder.train()
    train_loss = 0

    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for images, captions in loop:
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        optimizer.zero_grad()                    # Clear previous gradients / 清空梯度

        # 1) Forward pass / 1) 前向传播
        features = encoder(images)
        outputs = decoder(features, captions)

        # 2) Shift targets: predict tokens 1..end given 0..end-1
        # 2) 标签错位：用 [0..end-1] 预测 [1..end]
        targets = captions[:, 1:]

        # CE loss expects (N, C) and (N,) shapes / 交叉熵需要展平
        loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.reshape(-1))

        # 3) Backward + gradient clip + step / 3) 反向传播 + 梯度裁剪 + 参数更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5)   # Prevent exploding grads / 防梯度爆炸
        optimizer.step()

        train_loss += loss.item()
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- VALIDATION PHASE --- / --- 验证阶段 ---
    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():                        # No gradients during validation / 无需梯度
        for images, captions in val_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            features = encoder(images)
            outputs = decoder(features, captions)

            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.reshape(-1))
            val_loss += loss.item()

    # Log statistics / 记录本轮指标
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"\n=> Epoch {epoch} Complete")
    print(f"   Avg Train Loss: {avg_train_loss:.4f}")
    print(f"   Avg Val Loss:   {avg_val_loss:.4f}")

elapsed_time = time.time() - start_time
print(f"Total training time: {elapsed_time:.2f} s")

# Save summary for this configuration / 把本组超参的结果存进汇总表
res_e2_dmodel.append({
    "Embedding size": embed_size,
    "Average train loss": avg_train_loss,
    "Average val loss": avg_val_loss,
    "Training time (s)": elapsed_time
})'''

NEW_SOURCES["8d501c03"] = r'''# Plot training/validation loss curves / 绘制训练与验证 loss 曲线
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(epochs, val_losses,   marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.show()'''

NEW_SOURCES["0c7f5f46-8d3a-4447-a720-449c3c78e12a"] = r'''def unnormalize_image(image):
    '''Reverse ImageNet normalization so the image can be displayed.
    撤销 ImageNet 标准化，便于可视化显示。"""
    image = image.clone().detach().cpu()         # Avoid touching the original / 拷贝到 CPU

    # ImageNet stats / ImageNet 标准化统计量
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean                   # x' = x * std + mean / 还原像素
    image = image.clamp(0, 1)                    # Clip to [0,1] for display / 截断到 [0,1]

    return image"""

NEW_SOURCES["a200c89a-03a2-4cdf-be00-15e36ab0f2ab"] = r'''# ============ Qualitative inspection on validation set ============
# ============ 验证集上的定性检查（看几个例子） ============

# 1) Eval mode / 1) 切换为评估模式
encoder.eval()
decoder.eval()

# 2) Grab one batch from val_loader / 2) 取一个验证 batch
images, captions = next(iter(val_loader))

# 3) Show at most 2 examples / 3) 至多展示 2 个
num_examples = min(2, len(images))

# Stats used by the un-normalization (kept for reference) / 标准化所用统计量
mean = np.array([0.485, 0.4461, 0.4039])
std  = np.array([0.2695, 0.2623, 0.2772])

for i in range(num_examples):
    sample_img = images[i]
    sample_cap = captions[i]

    # 4) Generate prediction with greedy decoding / 4) 用贪心解码生成 caption
    prediction = generate_caption(encoder, decoder, sample_img, vocab)

    # 5) Build ground-truth string (drop specials) / 5) 还原参考 caption（去掉特殊 token）
    ground_truth = [vocab.itos[idx.item()] for idx in sample_cap
                    if vocab.itos[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]]
    ground_truth = ' '.join(ground_truth)

    # 6) Un-normalize + show / 6) 反标准化后显示
    img = unnormalize_image(image).permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        "Reference: " + " ".join(reference_caption) + "\n" +
        "Greedy: " + " ".join(greedy_caption) + "\n" +
        "Beam: " + " ".join(beam_caption),
        fontsize=10
    )
    plt.show()'''

NEW_SOURCES["oak-vNao0PZX"] = r'''# Evaluate the custom CNN + Transformer baseline / 评估自定义 CNN + Transformer 基线
metrics = evaluate_caption_metrics(
    encoder=encoder,
    decoder=decoder,
    val_dataset=val_dataset,
    vocab=train_dataset.vocab,
    max_samples=500,                             # Eval on 500 val samples / 在 500 条验证样本上评估
    decoding_method="greedy"                     # Greedy decoding / 贪心解码
)

# Pretty-print each metric / 打印每个指标
for metric_name, score in metrics.items():
    print(f"{metric_name}: {score:.4f}")'''

NEW_SOURCES["acc116a1"] = r'''# Torchvision ResNet imports / 引入 torchvision 中的 ResNet
import torchvision.models as models
from torchvision.models import ResNet18_Weights'''

NEW_SOURCES["9455073c"] = r'''class ResNetEncoder(nn.Module):
    '''Pretrained ResNet18 encoder. 预训练 ResNet18 编码器。
    Reuses ImageNet features and projects them to embed_size.
    复用 ImageNet 预训练特征，再投影到 embed_size。"""
    def __init__(self, embed_size, train_cnn=False):
        super(ResNetEncoder, self).__init__()

        # Load pretrained weights / 加载预训练权重
        weights = ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        # Drop the original 1000-class fc head, keep everything up to avgpool
        # 去掉 1000 类分类头，保留 avgpool 之前的全部模块
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone by default (only fine-tune if train_cnn=True)
        # 默认冻结 ResNet 参数，仅当 train_cnn=True 时微调
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

        # ResNet18 outputs 512-dim features; project to embed_size
        # ResNet18 输出 512 维特征，再线性映射到 embed_size
        self.fc = nn.Linear(512, embed_size)

        # Normalization + dropout for the projected vector / 归一化 + dropout
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        """images: (batch, 3, H, W) -> (batch, embed_size)"""

        # Backbone features / 主干网络特征 (B, 512, 1, 1)
        features = self.resnet(images)

        # Flatten spatial dims / 展平成 (B, 512)
        features = features.view(features.size(0), -1)

        # Project to embed_size / 投影到 embed_size
        features = self.fc(features)

        # Normalize + dropout / 归一化 + 随机失活
        features = self.bn(features)
        features = self.dropout(features)

        return features"""

NEW_SOURCES["8bfb09b6"] = r'''from torchvision import transforms
# Transform pipeline tuned for ImageNet-pretrained models
# ImageNet 预训练模型使用的标准预处理流程
transform_Res = transforms.Compose([
    transforms.Resize((224, 224)),                       # ResNet input size / ResNet 输入尺寸
    transforms.ToTensor(),                               # PIL -> [0,1] tensor / 转张量
    transforms.Normalize(                                # ImageNet mean/std / ImageNet 均值方差
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])'''

NEW_SOURCES["060f83e4"] = r'''# Train/val datasets using ImageNet-normalized transforms (for ResNet)
# 使用 ImageNet 标准化的训练/验证集（配合 ResNet 编码器）
train_dataset_res =  FlickrDataset(
    root_dir = data_location+"/Flicker8k_Dataset",
    captions_file = data_location+"/Flickr8k_text/Flickr8k.token.txt",
    transform=transform_Res
)

val_dataset_res =  FlickrDataset(
    root_dir = data_location+"/Flicker8k_Dataset",
    captions_file = data_location+"/Flickr8k_text/Flickr8k.token.txt",
    train=False,
    transform=transform_Res
)'''

NEW_SOURCES["6af8c056"] = r'''# ============ Training script: ResNet18 + Transformer ============
# ============ ResNet18 + Transformer 训练脚本 ============

IMAGE_SIZE = 224
BATCH_SIZE = 32

embed_size = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = train_dataset_res.vocab                  # Reuse shared vocab / 复用同一份词表
VOCAB_SIZE = len(vocab)
PAD_IDX = vocab.stoi["<PAD>"]

res_e2_dmodel = []
seed = 6421                                      # Reproducibility / 复现性
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Build encoder/decoder / 构建编码器与解码器
encoder = ResNetEncoder(
    embed_size=embed_size,
    train_cnn=False                              # Freeze ResNet backbone / 冻结 ResNet
).to(DEVICE)
decoder = DecoderTransformer(embed_size, VOCAB_SIZE).to(DEVICE)

# Loss with PAD ignored / 忽略 <PAD> 的交叉熵
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Only train trainable params: encoder.fc + encoder.bn + entire decoder
# 只训练 encoder 的投影层/归一化层和整个 decoder
params = list(encoder.fc.parameters()) + \
         list(encoder.bn.parameters()) + \
         list(decoder.parameters())

optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

start_time = time.time()


train_losses = []
val_losses = []


# Sanity check: encode one batch and print shapes / 先跑一个 batch 看看输出维度
image_batch, caption_batch = next(iter(train_loader))
image_batch = image_batch.to(DEVICE)

features = encoder(image_batch)

print("Image batch shape:", image_batch.shape)
print("Encoded feature shape:", features.shape)

for epoch in range(1, NUM_EPOCHS + 1):
    # --- TRAINING PHASE --- / --- 训练阶段 ---
    encoder.train()
    decoder.train()
    train_loss = 0

    loop = tqdm(train_loader, total=len(train_loader), leave=True)
    for images, captions in loop:
        images, captions = images.to(DEVICE), captions.to(DEVICE)

        optimizer.zero_grad()

        # Forward / 前向
        features = encoder(images)
        outputs = decoder(features, captions)

        # Shifted targets for teacher forcing / 教师强制目标
        targets = captions[:, 1:]

        loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.reshape(-1))

        # Backward + gradient clip / 反向 + 梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5)
        optimizer.step()

        train_loss += loss.item()
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- VALIDATION PHASE --- / --- 验证阶段 ---
    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)

            features = encoder(images)
            outputs = decoder(features, captions)

            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.reshape(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"\n=> Epoch {epoch} Complete")
    print(f"   Avg Train Loss: {avg_train_loss:.4f}")
    print(f"   Avg Val Loss:   {avg_val_loss:.4f}")

elapsed_time = time.time() - start_time
print(f"Total training time: {elapsed_time:.2f} s")

res_e2_dmodel.append({
    "Embedding size": embed_size,
    "Average train loss": avg_train_loss,
    "Average val loss": avg_val_loss,
    "Training time (s)": elapsed_time
})'''

NEW_SOURCES["83683f46"] = r'''# Plot ResNet-Transformer loss curves / 绘制 ResNet-Transformer 的 loss 曲线
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(epochs, val_losses,   marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.show()'''

NEW_SOURCES["690a849b"] = r'''# ============ Validation qualitative inspection (ResNet model) ============
# ============ 验证集定性检查（ResNet 版本） ============
encoder.eval()
decoder.eval()

# Take one batch from val_loader / 取一个验证 batch
images, captions = next(iter(val_loader))

# Show at most 2 examples / 最多展示 2 张
num_examples = min(2, len(images))

# Stats kept here for reference / 标准化统计量（仅作参考）
mean = np.array([0.485, 0.4461, 0.4039])
std  = np.array([0.2695, 0.2623, 0.2772])

for i in range(num_examples):
    sample_img = images[i]
    sample_cap = captions[i]

    # Greedy prediction / 贪心解码生成 caption
    prediction = generate_caption(encoder, decoder, sample_img, vocab)

    # Reference caption (drop specials) / 参考 caption（去除特殊 token）
    ground_truth = [vocab.itos[idx.item()] for idx in sample_cap
                    if vocab.itos[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]]
    ground_truth = ' '.join(ground_truth)

    img = unnormalize_image(image).permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        "Reference: " + " ".join(reference_caption) + "\n" +
        "Greedy: " + " ".join(greedy_caption) + "\n" +
        "Beam: " + " ".join(beam_caption),
        fontsize=10
    )
    plt.show()'''

NEW_SOURCES["148e5f16"] = r'''# Compare greedy vs beam search on one sample / 在单张图上对比贪心 vs 束搜索
image, caption = val_dataset_res[0]

# Greedy decoding / 贪心解码
greedy_caption = generate_caption(
    encoder=encoder,
    decoder=decoder,
    image=image,
    vocab=train_dataset_res.vocab,
    max_length=20
)

# Beam search decoding / 束搜索解码
beam_caption = generate_caption_beam_search(
    encoder=encoder,
    decoder=decoder,
    image=image,
    vocab=train_dataset_res.vocab,
    beam_size=3,
    max_length=20,
    length_penalty=0.7
)

# Ground-truth caption tokens / 参考 caption 的 token 列表
reference_caption = caption_tensor_to_tokens(
    caption,
    train_dataset_res.vocab
)

print("Reference:", " ".join(reference_caption))
print("Greedy:   ", " ".join(greedy_caption))
print("Beam:     ", " ".join(beam_caption))'''

NEW_SOURCES["19b4ae59"] = r'''import matplotlib.pyplot as plt

# Pick a sample / 取一张样本
image, caption = val_dataset_res[0]

# Greedy / 贪心结果
greedy_caption = generate_caption(
    encoder=encoder,
    decoder=decoder,
    image=image,
    vocab=train_dataset_res.vocab,
    max_length=20
)

# Beam search / 束搜索结果
beam_caption = generate_caption_beam_search(
    encoder=encoder,
    decoder=decoder,
    image=image,
    vocab=train_dataset_res.vocab,
    beam_size=3,
    max_length=20,
    length_penalty=0.7
)

# Reference caption / 参考 caption
reference_caption = caption_tensor_to_tokens(
    caption,
    train_dataset_res.vocab
)

# Un-normalize the image for display / 反标准化以便显示
img = unnormalize_image(image).permute(1, 2, 0).numpy()

# Plot image with all 3 captions in the title / 把三条 caption 放在标题中显示
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.title(
    "Reference: " + " ".join(reference_caption) + "\n" +
    "Greedy: " + " ".join(greedy_caption) + "\n" +
    "Beam: " + " ".join(beam_caption),
    fontsize=10
)
plt.show()'''

NEW_SOURCES["e6709915"] = r'''# Greedy-decoded BLEU/ROUGE-L for ResNet+Transformer / 贪心解码的 BLEU/ROUGE-L
resnet_greedy_metrics = evaluate_caption_metrics(
    encoder=encoder,
    decoder=decoder,
    val_dataset=val_dataset_res,
    vocab=train_dataset_res.vocab,
    max_samples=500,
    decoding_method="greedy"
)

# Beam-decoded BLEU/ROUGE-L (usually much higher) / 束搜索的指标，通常显著更高
resnet_beam_metrics = evaluate_caption_metrics(
    encoder=encoder,
    decoder=decoder,
    val_dataset=val_dataset_res,
    vocab=train_dataset_res.vocab,
    max_samples=500,
    decoding_method="beam",
    beam_size=3,
    length_penalty=0.7
)



print("ResNet18 + Greedy Decoding")
for metric_name, score in resnet_greedy_metrics.items():
    print(f"{metric_name}: {score:.4f}")

print("\nResNet18 + Beam Search")
for metric_name, score in resnet_beam_metrics.items():
    print(f"{metric_name}: {score:.4f}")'''

NEW_SOURCES["8b9d1e61"] = r'''# Side-by-side comparison of greedy vs beam metrics / 把贪心和束搜索结果并排展示
decoding_comparison_df = pd.DataFrame({
    "ResNet18 + Greedy": resnet_greedy_metrics,
    "ResNet18 + Beam Search": resnet_beam_metrics
})

decoding_comparison_df'''

NEW_SOURCES["7e3fd43f-c6e4-425e-918e-6e7162d827cf"] = r'''def collect_caption_examples(
    encoder,
    decoder,
    val_dataset,
    vocab,
    num_examples=20,
    decoding_method="beam",
    beam_size=3,
    length_penalty=0.7
):
    '''Collect (reference, prediction) pairs for manual error analysis.
    收集 (参考, 预测) 样本，便于人工做错误分析。"""
    examples = []

    encoder.eval()
    decoder.eval()

    for i in range(num_examples):
        image, caption = val_dataset[i]

        # Greedy branch / 贪心分支
        if decoding_method == "greedy":
            predicted_caption = generate_caption(
                encoder=encoder,
                decoder=decoder,
                image=image,
                vocab=vocab,
                max_length=20
            )

        # Beam search branch / 束搜索分支
        elif decoding_method == "beam":
            predicted_caption = generate_caption_beam_search(
                encoder=encoder,
                decoder=decoder,
                image=image,
                vocab=vocab,
                beam_size=beam_size,
                max_length=20,
                length_penalty=length_penalty
            )

        else:
            raise ValueError("decoding_method must be 'greedy' or 'beam'.")

        # Decode reference caption tensor / 还原参考 caption 的文本
        reference_caption = caption_tensor_to_tokens(caption, vocab)

        examples.append({
            "index": i,
            "reference": " ".join(reference_caption),
            "prediction": " ".join(predicted_caption)
        })

    return examples
    """

NEW_SOURCES["0779734e-62d7-434d-aedb-684e95c34efb"] = r'''# Run beam-search decoding on the first 20 val samples / 在前 20 条验证样本上跑束搜索
beam_examples = collect_caption_examples(
    encoder=encoder,
    decoder=decoder,
    val_dataset=val_dataset_res,
    vocab=train_dataset_res.vocab,
    num_examples=20,
    decoding_method="beam",
    beam_size=3,
    length_penalty=0.7
)

# Put results into a DataFrame for inspection / 整理成 DataFrame 方便查看
beam_examples_df = pd.DataFrame(beam_examples)
beam_examples_df'''

NEW_SOURCES["f3b1f349-e126-4978-b9e9-3ec7bd31f9ec"] = r'''# Add empty columns for human-labelled error type / 添加人工标注列
# Fill in error_type later by visual inspection / 之后人工逐条填写错误类型
beam_examples_df["error_type"] = ""
beam_examples_df["notes"] = ""

beam_examples_df'''

NEW_SOURCES["c342c357-be9b-419e-913f-9f7dd8bf2721"] = r'''from collections import Counter

# Aggregate counts of each error label / 统计每种错误类型的出现次数
error_counter = Counter()

for labels in beam_examples_df["error_type"]:
    if labels.strip() == "":                     # Skip unlabelled rows / 跳过未标注行
        continue

    # One row may have multiple labels separated by ";"
    # 一行可能有多个标签，用 ";" 分隔
    for label in labels.split(";"):
        error_counter[label.strip()] += 1

# Summary DataFrame sorted by count / 按次数降序的汇总表
error_summary_df = pd.DataFrame(
    error_counter.items(),
    columns=["Error Type", "Count"]
).sort_values(by="Count", ascending=False)

error_summary_df'''

NEW_SOURCES["90e38208-ea69-4645-8c9a-af5b19795291"] = r'''def show_error_examples(
    examples_df,
    val_dataset,
    num_examples=5
):
    '''Display image + reference + prediction + error_type for inspection.
    可视化几条样本：图像 + 参考 + 预测 + 错误类型。"""
    for row_idx in range(min(num_examples, len(examples_df))):
        sample = examples_df.iloc[row_idx]
        # Use the saved val-dataset index to retrieve the image / 用记录的索引取回图像
        image, _ = val_dataset[int(sample["index"])]

        img = unnormalize_image(image).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")

        # Title shows ref + pred + label / 标题展示参考/预测/错误类型
        title = (
            f"Reference: {sample['reference']}\n"
            f"Prediction: {sample['prediction']}\n"
            f"Error Type: {sample['error_type']}"
        )

        plt.title(title, fontsize=9)
        plt.show()"""

NEW_SOURCES["8268f776-7e8b-40ae-b0d9-54422d43f859"] = r'''# Show the first 5 examples with their error labels / 展示前 5 条错误分析样本
show_error_examples(
    examples_df=beam_examples_df,
    val_dataset=val_dataset_res,
    num_examples=5
)'''


def to_source_lines(s: str):
    """Split a source string into the per-line list nbformat expects.
    Each line except the last keeps its trailing newline."""
    lines = s.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + "\n")
        else:
            if line != "":
                out.append(line)
    return out


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    updated = 0
    missed = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id")
        if cid in NEW_SOURCES:
            cell["source"] = to_source_lines(NEW_SOURCES[cid])
            updated += 1
        else:
            # Skip empty cells silently
            src = cell.get("source")
            if src and (isinstance(src, list) and "".join(src).strip()) or (isinstance(src, str) and src.strip()):
                missed.append(cid)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {updated} cells.")
    if missed:
        print("Code cells with no mapping (left unchanged):")
        for m in missed:
            print(" -", m)


if __name__ == "__main__":
    main()
