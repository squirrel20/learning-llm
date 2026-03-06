"""
Token Embedding 训练演示 — Mini GPT Transformer

核心思想:
  Embedding（嵌入）是将离散的 token id 映射为连续的稠密向量。
  这些向量在训练过程中不断更新，使得语义相近的 token 在向量空间中距离更近。

  本脚本通过一个小型 GPT Transformer 模型来训练 nn.Embedding 层。
  相比 Bigram 模型（只看 1 个 token 的上下文），Transformer 使用自注意力机制
  让每个位置都能"看到"之前所有位置的 token，从而在更丰富的上下文中训练 embedding。

  模型架构参考 GPT：
    - 可学习的 token embedding + 位置 embedding
    - 多层 Transformer Block（多头自注意力 + 前馈网络 + 残差连接 + LayerNorm）
    - 因果注意力掩码（causal mask）确保只看过去，不偷看未来

  训练完成后，用 PCA（主成分分析）将高维 embedding 降到 2 维进行可视化，
  观察哪些 token 在向量空间中聚集在一起。

流程:
  1. 加载之前训练好的 BPE tokenizer，对语料分词
  2. 构建 Mini GPT Transformer 模型
  3. 在语料上训练模型
  4. PCA 降维可视化 embedding 空间
  5. 探针：对测试提示词展示 分词 → 嵌入 → 相似度 → 上下文预测 → 自回归生成
"""

import sys
from pathlib import Path

import matplotlib
# 使用非交互式后端，这样在没有 GUI 环境（如服务器）上也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA  # 用于将高维 embedding 降维到 2D 可视化

# 把 01-basics 目录加到 sys.path，方便 import bpe_tokenizer 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01-basics"))
from bpe_tokenizer import ByteLevelBPETokenizer


# ============================================================
# 1. 加载 BPE tokenizer & 对语料分词
# ============================================================

def load_data():
    """加载预训练的 BPE tokenizer，并将整个语料文本编码为 token id 序列。

    返回:
        tokenizer: BPE 分词器实例（包含词表和合并规则）
        token_ids: 整个语料编码后的 token id 列表，如 [102, 3, 455, 12, ...]
    """
    data_dir = Path(__file__).resolve().parent.parent / "01-basics" / "data"
    tokenizer = ByteLevelBPETokenizer()
    # 加载之前用 bpe_tokenizer.py 训练好的 BPE 模型（词表 + 合并规则）
    tokenizer.load(str(data_dir / "bpe_model.json"))

    # 读取语料并编码为 token id 序列
    corpus_text = (data_dir / "corpus.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(corpus_text)
    print(f"语料 token 数: {len(token_ids):,}")
    return tokenizer, token_ids


# ============================================================
# 2. Mini GPT Transformer 模型
# ============================================================
#
# GPT 架构的核心是 Transformer Decoder（只使用因果自注意力）。
# 与 Bigram 模型不同，这里每个 token 可以关注它之前的所有 token，
# 从而利用更丰富的上下文信息来预测下一个 token。
#
# 模型结构:
#   token_id → token_embedding + position_embedding
#           → N 层 TransformerBlock（自注意力 + FFN + 残差 + LayerNorm）
#           → LayerNorm → Linear(lm_head) → vocab_size 维 logits


class CausalSelfAttention(nn.Module):
    """多头因果自注意力（Multi-Head Causal Self-Attention）

    自注意力的核心思想：对序列中每个位置，计算它与所有其他位置的"相关程度"（注意力权重），
    然后用这些权重对所有位置的值（Value）做加权求和，得到融合了上下文信息的新表示。

    "因果"（causal）意味着每个位置只能关注它自己和之前的位置，不能偷看未来。
    这是通过一个下三角矩阵掩码（tril mask）实现的。

    "多头"（multi-head）是把 embedding 维度拆分成多个"头"，每个头独立计算注意力，
    最后拼接起来。不同的头可以关注不同类型的模式（如语法关系、语义关联等）。
    """
    def __init__(self, embed_dim: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads  # 每个注意力头的维度

        # 一次性计算 Q、K、V 三个投影矩阵，效率比分开计算更高
        # Q（Query）: 我在找什么？ K（Key）: 我有什么？ V（Value）: 我提供什么信息？
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果掩码：下三角矩阵，确保位置 i 只能关注位置 0..i
        # register_buffer 让它成为模型状态的一部分（会跟随 .to(device)），但不参与梯度计算
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch_size, seq_len, embed_dim

        # 一次性计算 Q, K, V 并拆分成多个头
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力: attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        # 除以 sqrt(d_k) 是为了防止点积值过大导致 softmax 梯度消失
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # 应用因果掩码：将未来位置的注意力分数设为 -inf，softmax 后变为 0
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 用注意力权重对 V 做加权求和
        out = attn @ v  # (B, n_heads, T, head_dim)
        # 把多个头的输出拼接回来
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """Transformer Block — 自注意力 + 前馈网络 + 残差连接 + LayerNorm

    采用 Pre-Norm 结构（先 LayerNorm 再做计算），这是 GPT-2 以来的标准做法，
    比原始 Transformer 的 Post-Norm 训练更稳定。

    残差连接（Residual Connection）: x = x + sublayer(x)
    让梯度可以"跳过"子层直接回传，解决深层网络的梯度消失问题。

    LayerNorm: 对每个样本的特征维度做归一化（均值=0, 方差=1），稳定训练。
    """
    def __init__(self, embed_dim: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        # 前馈网络（FFN）: 两层线性变换 + GELU 激活
        # 隐藏层是 4 倍 embed_dim，这是 Transformer 的惯例
        # GELU 是比 ReLU 更平滑的激活函数，GPT 系列模型都使用它
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-Norm 残差结构
        x = x + self.attn(self.ln1(x))  # 注意力子层 + 残差
        x = x + self.ffn(self.ln2(x))   # FFN 子层 + 残差
        return x


class MiniGPT(nn.Module):
    """Mini GPT — 简化版 GPT 语言模型

    结构:
      1. Token Embedding: 将 token id 映射为 embed_dim 维向量
      2. Position Embedding: 可学习的位置编码，让模型知道 token 的位置信息
         （因为自注意力本身是位置无关的，必须显式注入位置信息）
      3. N 层 TransformerBlock: 通过自注意力融合上下文
      4. LayerNorm + Linear(lm_head): 输出 vocab_size 维的 logits
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, n_heads: int = 4,
                 n_layers: int = 4, block_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size

        # Token embedding: 每个 token id → embed_dim 维向量
        # 这就是我们最终要可视化和分析的 embedding 层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置 embedding: 序列中每个位置 → embed_dim 维向量
        # 位置 0, 1, 2, ... 各有独立的可学习向量
        self.pos_embedding = nn.Embedding(block_size, embed_dim)

        self.drop = nn.Dropout(dropout)
        # N 层 Transformer Block 堆叠
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)  # 最终的 LayerNorm
        # Language Model Head: 将 embed_dim 维的隐藏状态映射回 vocab_size 维的 logits
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # 打印参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {n_params:,}")

    def forward(self, idx):
        # idx: (B, T) — 一批 token id 序列
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"

        # 位置索引: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=idx.device)

        # Token embedding + 位置 embedding
        # 两者相加让模型同时知道"是什么 token"和"在什么位置"
        tok_emb = self.embedding(idx)      # (B, T, embed_dim)
        pos_emb = self.pos_embedding(pos)  # (T, embed_dim) — 自动广播到 batch 维
        x = self.drop(tok_emb + pos_emb)

        # 通过 N 层 Transformer Block
        x = self.blocks(x)         # (B, T, embed_dim)
        x = self.ln_f(x)           # 最终 LayerNorm
        logits = self.lm_head(x)   # (B, T, vocab_size) — 每个位置预测下一个 token
        return logits


# ============================================================
# 3. 训练
# ============================================================

# 自动选择可用的最佳计算设备：Apple Silicon GPU (mps) > NVIDIA GPU (cuda) > CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    """从语料中随机采样一个 mini-batch 的训练样本。

    每个样本是一段长度为 block_size 的连续 token 序列。
    输入 x 是前 block_size 个 token，标签 y 是后移一位的 block_size 个 token。

    例如 block_size=4, 从位置 10 开始:
      x = [t10, t11, t12, t13]    （输入）
      y = [t11, t12, t13, t14]    （标签）
    模型需要在每个位置预测下一个 token。
    """
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


def train(token_ids: list[int], vocab_size: int, embed_dim: int = 64,
          block_size: int = 64, n_iters: int = 5000, batch_size: int = 32,
          lr: float = 3e-4):
    """在语料上训练 Mini GPT 模型，学习 token embedding。

    与 Bigram 模型的关键区别:
      - 每个训练样本是一段长度为 block_size 的连续序列（而非单个 token 对）
      - 模型可以利用之前所有 token 的上下文来预测下一个 token
      - 使用迭代次数而非 epoch 来控制训练（因为每次随机采样，不需要遍历全部数据）

    参数:
        token_ids: 语料编码后的 token id 序列
        vocab_size: 词表大小
        embed_dim: embedding 向量维度
        block_size: 上下文窗口大小
        n_iters: 训练迭代次数
        batch_size: 每批序列数
        lr: 学习率
    """
    print(f"使用设备: {DEVICE}")

    model = MiniGPT(vocab_size, embed_dim, block_size=block_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 交叉熵损失：衡量模型输出的概率分布与真实标签之间的差异
    criterion = nn.CrossEntropyLoss()

    data = torch.tensor(token_ids, dtype=torch.long)

    model.train()
    for step in range(1, n_iters + 1):
        # 随机采样一个 mini-batch
        x_batch, y_batch = get_batch(data, block_size, batch_size, DEVICE)

        # 前向传播
        logits = model(x_batch)  # (B, T, vocab_size)
        # 展平为 (B*T, vocab_size) 和 (B*T,) 以计算交叉熵
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))

        # 反向传播 & 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 500 步打印一次 loss
        if step % 500 == 0 or step == 1:
            print(f"Step {step:5d}/{n_iters}  loss: {loss.item():.4f}")

    return model


# ============================================================
# 4. 可视化
# ============================================================
#
# 将训练好的 embedding 向量用 PCA 降维到 2 维平面上绘制散点图。
# PCA（主成分分析）会找到数据中方差最大的两个方向作为新坐标轴，
# 从而在尽量保留信息的前提下将高维向量压缩到 2 维。
#
# 在可视化图中，语义或功能相近的 token 应该聚集在一起，
# 例如：数字聚在一起、标点聚在一起、常见英文子词聚在一起等。

def visualize(model: MiniGPT, tokenizer: ByteLevelBPETokenizer, save_path: str):
    """将 embedding 权重矩阵用 PCA 降维并绘制 2D 散点图。"""
    # 取出 embedding 层的权重矩阵，shape = (vocab_size, embed_dim)
    # detach() 断开计算图，cpu() 移到 CPU，numpy() 转为 NumPy 数组供 sklearn 使用
    embeddings = model.embedding.weight.detach().cpu().numpy()

    # PCA 降到 2 维
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)  # shape = (vocab_size, 2)
    # explained_variance_ratio_ 表示两个主成分解释了原始数据多少比例的方差
    # 比例越高，说明 2D 图保留的信息越多
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # 选出要标注的 token（不可能标注所有 token，会太拥挤）
    labels_to_show = {}

    # 标注所有可见 ASCII 字符（id 0-255 对应单字节，是 byte-level BPE 的基础字符）
    for tid in range(256):
        b = tokenizer.vocab[tid]
        if len(b) == 1 and 33 <= b[0] <= 126:  # 可打印 ASCII（排除空格和控制字符）
            labels_to_show[tid] = chr(b[0])

    # 标注 BPE 合并产生的前 200 个子词 token
    # 最早合并的 token 对应语料中最高频的字节对，通常是最常见的子词（如 "th", "he", "in" 等）
    for pair, new_id in tokenizer.merges[:200]:
        try:
            text = tokenizer.vocab[new_id].decode("utf-8")
            if text.strip():  # 跳过纯空白 token
                labels_to_show[new_id] = text
        except UnicodeDecodeError:
            pass  # 跳过无法解码为有效 UTF-8 的 token

    # 绘图
    plt.figure(figsize=(16, 12))
    # 先画所有 token 的散点（小点、半透明，作为背景）
    plt.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.3, c="steelblue")

    # 在选中的 token 位置标注文本
    for tid, label in labels_to_show.items():
        if tid < len(coords):
            plt.annotate(label, (coords[tid, 0], coords[tid, 1]),
                         fontsize=6, alpha=0.7,
                         fontfamily="sans-serif")

    plt.title("Token Embedding Space (PCA)")
    plt.xlabel("PC1")  # 第一主成分方向
    plt.ylabel("PC2")  # 第二主成分方向
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"可视化已保存: {save_path}")


# ============================================================
# 5. 探针：测试一个提示词的分词 + 嵌入效果
# ============================================================
#
# 这个函数用于直观地展示一个输入文本在模型中经历的完整过程：
#   文本 → BPE 分词 → token id 序列 → embedding 向量 → 相似度分析
#        → Transformer 上下文预测 → 自回归生成
# 这有助于理解 LLM 处理文本的前几个步骤。

def probe_prompt(prompt: str, model: MiniGPT, tokenizer: ByteLevelBPETokenizer):
    """展示一个提示词经过 BPE 分词 → Embedding → Transformer 推理的完整过程。"""
    print(f"\n{'='*60}")
    print(f"探针输入: {prompt!r}")
    print(f"{'='*60}")

    # --- 第一步：BPE 分词 ---
    # 将输入文本编码为 token id 序列
    token_ids = tokenizer.encode(prompt)
    # 将每个 token id 解码回可读文本，方便展示
    tokens_text = []
    for tid in token_ids:
        try:
            t = tokenizer.vocab[tid].decode("utf-8")
        except UnicodeDecodeError:
            t = str(tokenizer.vocab[tid])  # 无法解码为 UTF-8 时显示字节表示
        tokens_text.append(t)

    print(f"\n[分词结果]  共 {len(token_ids)} 个 token")
    for i, (tid, text) in enumerate(zip(token_ids, tokens_text)):
        print(f"  {i:3d} │ id={tid:5d} │ {text!r}")

    # --- 第二步：查表获取 embedding 向量 ---
    model.eval()
    with torch.no_grad():
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        # 直接用 embedding 层查表，得到每个 token 对应的 64 维向量
        emb = model.embedding(ids_tensor).cpu().numpy()  # (seq_len, embed_dim)

    # 打印每个 token 的 embedding 向量（只显示前 8 维）
    print(f"\n[Embedding]  shape = {emb.shape}")
    for i, (tid, text) in enumerate(zip(token_ids, tokens_text)):
        vec_str = "  ".join(f"{v:+.3f}" for v in emb[i, :8])
        print(f"  {i:3d} │ {text!r:12s} │ [{vec_str}  ...]")

    # --- 第三步：计算 token 间的余弦相似度 ---
    if len(token_ids) >= 2:
        emb_t = torch.tensor(emb)
        norms = emb_t / emb_t.norm(dim=1, keepdim=True)
        sim_matrix = (norms @ norms.T).numpy()

        print(f"\n[余弦相似度矩阵]")
        header = "        " + "".join(f"{i:7d}" for i in range(len(token_ids)))
        print(header)
        for i in range(len(token_ids)):
            row = f"  {i:3d}   " + "".join(f"{sim_matrix[i,j]:+.3f} " for j in range(len(token_ids)))
            print(row)

    # --- 第四步：Transformer 上下文预测 ---
    # 与 Bigram 不同，这里输入完整序列，每个位置的预测都基于它之前所有 token 的上下文
    print(f"\n[Transformer 上下文预测: 每个位置 → top-3 下一个 token]")
    with torch.no_grad():
        # 输入 (1, T) 的序列，输出 (1, T, vocab_size) 的 logits
        input_ids = ids_tensor.unsqueeze(0)  # (1, T)
        logits = model(input_ids)  # (1, T, vocab_size)
        probs = torch.softmax(logits[0], dim=-1)  # (T, vocab_size)
        top3 = torch.topk(probs, k=3, dim=-1)

    for i, (tid, text) in enumerate(zip(token_ids, tokens_text)):
        context = " ".join(tokens_text[:i+1])
        preds = []
        for j in range(3):
            pred_id = top3.indices[i, j].item()
            pred_prob = top3.values[i, j].item()
            try:
                pred_text = tokenizer.vocab[pred_id].decode("utf-8")
            except UnicodeDecodeError:
                pred_text = str(tokenizer.vocab[pred_id])
            preds.append(f"{pred_text!r}({pred_prob:.1%})")
        print(f"  [{context}] → {', '.join(preds)}")

    # --- 第五步：自回归生成 ---
    # 从给定的 prompt 开始，逐个 token 生成后续内容
    # 这就是 GPT 生成文本的核心方式：每次用模型预测下一个 token，加到序列末尾，重复
    print(f"\n[自回归生成]  续写 30 个 token:")
    generated = list(token_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            # 取最近 block_size 个 token 作为输入（超出窗口的截掉）
            context = generated[-model.block_size:]
            x = torch.tensor([context], dtype=torch.long, device=DEVICE)
            logits = model(x)  # (1, T, vocab_size)
            # 只取最后一个位置的预测（即对下一个 token 的预测）
            next_logits = logits[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(next_logits, dim=-1)
            # 按概率采样（而非贪心取 argmax），让生成更多样
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)

    # 将生成的 token id 序列解码回文本
    generated_bytes = b"".join(tokenizer.vocab[tid] for tid in generated)
    try:
        generated_text = generated_bytes.decode("utf-8", errors="replace")
    except Exception:
        generated_text = str(generated_bytes)
    print(f"  {generated_text}")


# ============================================================
# main — 串联整个流程
# ============================================================

if __name__ == "__main__":
    # 第 1 步：加载 BPE tokenizer 并将语料编码为 token id 序列
    tokenizer, token_ids = load_data()
    vocab_size = len(tokenizer.vocab)
    print(f"词表大小: {vocab_size}")

    # 第 2 步：训练 Mini GPT 模型，学习 token embedding
    block_size = 64
    model = train(token_ids, vocab_size, block_size=block_size)

    # 第 3 步：PCA 降维可视化 embedding 空间
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    save_path = str(output_dir / "embedding_viz.png")
    visualize(model, tokenizer, save_path)

    # 第 4 步：用探针函数展示几个测试提示词的完整处理过程
    test_prompts = [
        "The meaning of life is",   # 英文句子
        "Hello World",               # 经典短语
    ]
    for prompt in test_prompts:
        probe_prompt(prompt, model, tokenizer)
