"""
Simple Self-Attention Implementation
从零理解 Self-Attention 机制

学习目标:
1. 理解 Q, K, V 的概念
2. 理解 Attention Score 的计算
3. 理解 Scaled Dot-Product Attention

推荐先阅读: Jay Alammar 的 "The Illustrated Transformer"
https://jalammar.github.io/illustrated-transformer/
"""

import numpy as np

def softmax(x, axis=-1):
    """Softmax 函数: 将分数转换为概率分布"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    参数:
        Q: Query 矩阵, shape (seq_len, d_k)
        K: Key 矩阵, shape (seq_len, d_k)
        V: Value 矩阵, shape (seq_len, d_v)
        mask: 可选的掩码矩阵 (用于 causal attention)

    返回:
        output: 注意力加权后的输出
        attention_weights: 注意力权重矩阵
    """
    d_k = Q.shape[-1]

    # Step 1: 计算 Q 和 K 的点积
    # 直觉: 衡量每个位置与其他位置的相关性
    scores = np.matmul(Q, K.T)  # shape: (seq_len, seq_len)

    # Step 2: 缩放 (除以 sqrt(d_k))
    # 原因: 防止点积值过大导致 softmax 梯度消失
    scores = scores / np.sqrt(d_k)

    # Step 3: 应用掩码 (可选, 用于 decoder 的 causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax 得到注意力权重
    attention_weights = softmax(scores, axis=-1)

    # Step 5: 加权求和 Value
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def create_causal_mask(seq_len):
    """
    创建因果掩码 (Causal Mask)
    用于防止 decoder 看到未来的 token

    例如 seq_len=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def demo_self_attention():
    """演示 Self-Attention 的工作原理"""
    print("=" * 60)
    print("Self-Attention 演示")
    print("=" * 60)

    # 假设我们有一个简单的序列: 3个token, 每个token的embedding维度是4
    seq_len = 3
    d_model = 4

    # 模拟输入序列的 embedding
    # 在实际中，这是词嵌入 + 位置编码
    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)

    print(f"\n输入序列 X (shape: {X.shape}):")
    print(X)

    # 在简化版本中，Q = K = V = X (无投影矩阵)
    # 实际 Transformer 中会有 Wq, Wk, Wv 投影矩阵
    Q = K = V = X

    # 计算 Self-Attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print(f"\n注意力权重矩阵 (shape: {attention_weights.shape}):")
    print(attention_weights)
    print("\n解读: 每行表示一个 token 对所有 token 的注意力分布")
    print("      所有权重加起来等于 1 (softmax 的结果)")

    print(f"\n输出 (shape: {output.shape}):")
    print(output)
    print("\n解读: 每个 token 的新表示是所有 token 的加权组合")


def demo_causal_attention():
    """演示 Causal (因果) Attention - GPT 使用的机制"""
    print("\n" + "=" * 60)
    print("Causal Attention 演示 (GPT 使用)")
    print("=" * 60)

    seq_len = 4
    d_model = 4

    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)
    Q = K = V = X

    # 创建因果掩码
    mask = create_causal_mask(seq_len)
    print(f"\n因果掩码 (shape: {mask.shape}):")
    print(mask)
    print("\n解读: 1 表示可以看到, 0 表示不能看到")
    print("      每个 token 只能看到它自己和之前的 token")

    # 使用因果掩码计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print(f"\n因果注意力权重矩阵:")
    print(np.round(attention_weights, 3))
    print("\n解读: 上三角部分都是 0 (看不到未来)")


def demo_multi_head_attention():
    """演示 Multi-Head Attention 的概念"""
    print("\n" + "=" * 60)
    print("Multi-Head Attention 概念")
    print("=" * 60)

    print("""
Multi-Head Attention 的核心思想:

1. 将 Q, K, V 分成多个 "head"
2. 每个 head 独立计算注意力
3. 最后拼接所有 head 的输出

为什么需要多个 head?
- 不同的 head 可以关注不同类型的关系
- 例如: 一个 head 关注语法关系, 另一个关注语义关系

公式:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

    其中 head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)
""")


if __name__ == "__main__":
    demo_self_attention()
    demo_causal_attention()
    demo_multi_head_attention()

    print("\n" + "=" * 60)
    print("下一步学习建议:")
    print("=" * 60)
    print("""
1. 阅读 "Attention Is All You Need" 论文
2. 观看 Andrej Karpathy 的 "Let's build GPT" 视频
3. 实现完整的 Multi-Head Attention
4. 实现完整的 Transformer Encoder/Decoder
""")
