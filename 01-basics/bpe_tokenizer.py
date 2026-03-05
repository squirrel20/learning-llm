"""
Byte-Level BPE 分词器 — 从零实现

核心思想:
- 初始词表 = 256 个字节 (0x00-0xFF)
- 反复合并最高频的相邻 pair，每次合并产生一个新 token
- 天然支持任何语言，因为一切文本最终都是字节序列

参考: GPT-2 论文 (Radford et al., 2019)
"""

import json


class ByteLevelBPETokenizer:
    def __init__(self):
        # merges: 按训练顺序记录的合并规则 [(pair, new_id), ...]
        self.merges: list[tuple[tuple[int, int], int]] = []
        # vocab: {token_id: bytes}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ----------------------------------------------------------------
    # 训练
    # ----------------------------------------------------------------

    def train(self, text: str, vocab_size: int = 1256, verbose: bool = False):
        """
        在文本上训练 BPE。

        Args:
            text: 训练文本
            vocab_size: 目标词表大小 (>= 256)
            verbose: 是否打印每一步的合并信息
        """
        assert vocab_size >= 256, "vocab_size 必须 >= 256（256 个基础字节 token）"
        num_merges = vocab_size - 256

        # 文本 → UTF-8 字节序列
        ids = list(text.encode("utf-8"))
        if verbose:
            print(f"原始字节长度: {len(ids):,}")

        for i in range(num_merges):
            # 统计相邻 pair 频率
            counts = _count_pairs(ids)
            if not counts:
                if verbose:
                    print(f"第 {i} 步: 没有更多可合并的 pair，提前停止")
                break

            # 找最高频 pair
            best_pair = max(counts, key=counts.get)
            new_id = 256 + i

            # 执行合并
            ids = _merge(ids, best_pair, new_id)

            # 记录
            self.merges.append((best_pair, new_id))
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i < 20 or i % 100 == 0):
                token_str = self.vocab[new_id]
                # 尝试解码为可读字符串
                try:
                    readable = token_str.decode("utf-8")
                except UnicodeDecodeError:
                    readable = str(token_str)
                print(
                    f"  合并 #{i:4d}: {best_pair} → {new_id}  "
                    f"freq={counts[best_pair]:,}  "
                    f"token={readable!r}  "
                    f"序列长度={len(ids):,}"
                )

        if verbose:
            ratio = len(text.encode("utf-8")) / len(ids) if ids else 0
            print(f"\n训练完成: 词表大小={len(self.vocab)}, 压缩比={ratio:.2f}x")

    # ----------------------------------------------------------------
    # 编码 / 解码
    # ----------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """将文本编码为 token ID 序列"""
        ids = list(text.encode("utf-8"))
        # 按训练时的顺序依次应用每条合并规则
        for pair, new_id in self.merges:
            ids = _merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码为文本"""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ----------------------------------------------------------------
    # 保存 / 加载
    # ----------------------------------------------------------------

    def save(self, path: str):
        """保存模型到 JSON 文件"""
        data = {
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """从 JSON 文件加载模型"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}

        for entry in data["merges"]:
            pair = (entry["pair"][0], entry["pair"][1])
            new_id = entry["new_id"]
            self.merges.append((pair, new_id))
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]


# ====================================================================
# 辅助函数
# ====================================================================

def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计相邻 pair 出现次数"""
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """将序列中所有出现的 pair 替换为 new_id"""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


# ====================================================================
# 快速自测
# ====================================================================

if __name__ == "__main__":
    tok = ByteLevelBPETokenizer()

    # 用小文本快速测试
    sample = "hello world hello world hello" * 100
    tok.train(sample, vocab_size=270, verbose=True)

    # 往返一致性测试
    test_texts = [
        "hello world",
        "你好世界",
        "Hello, 世界! 🌍",
        "",
    ]
    print("\n往返一致性测试:")
    for t in test_texts:
        ids = tok.encode(t)
        decoded = tok.decode(ids)
        status = "PASS" if decoded == t else "FAIL"
        print(f"  [{status}] {t!r} → {len(ids)} tokens → {decoded!r}")
