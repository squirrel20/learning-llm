# 01-basics: Tokenization 与 Byte-Level BPE

训练 nano GPT 的第一步 — 理解并实现分词器。

## 文件结构

```
01-basics/
├── bpe_tokenizer.py          # BPE 核心实现（纯标准库，零依赖）
├── bpe_train_and_test.ipynb  # 交互式训练 + 可视化 notebook
├── tokenization_demo.py      # 分词概念演示（tiktoken / sentencepiece）
├── prepare_corpus.py         # 语料下载脚本
└── data/
    ├── corpus.txt            # 训练语料（中英混合，~1.3MB）
    └── bpe_model.json        # 训练好的 BPE 模型文件
```

## 快速开始

### 1. 准备语料

```bash
uv run python 01-basics/prepare_corpus.py
```

下载 Tiny Shakespeare（英文）并生成中文文本，合并为 `data/corpus.txt`。

### 2. 训练分词器

```python
from bpe_tokenizer import ByteLevelBPETokenizer

# 加载语料
with open("data/corpus.txt", "r") as f:
    text = f.read()

# 训练：256 基础字节 + 1000 次合并 = 1256 词表
tok = ByteLevelBPETokenizer()
tok.train(text, vocab_size=1256, verbose=True)

# 保存模型
tok.save("data/bpe_model.json")
```

### 3. 编码 / 解码

```python
tok = ByteLevelBPETokenizer()
tok.load("data/bpe_model.json")

# 编码
ids = tok.encode("Hello, 世界!")
print(ids)  # [72, 490, 261, 741, 33]

# 解码
text = tok.decode(ids)
print(text)  # Hello, 世界!

# 往返一致性：对任意文本都成立
assert tok.decode(tok.encode("你好世界 🌍")) == "你好世界 🌍"
```

### 4. 交互式 Notebook

```bash
uv run jupyter notebook 01-basics/bpe_train_and_test.ipynb
```

Notebook 包含 9 个部分：

1. UTF-8 字节基础 — 中英文字节编码差异
2. 加载语料 + 转字节序列
3. 统计 pair 频率 — Top-10 pair 分析
4. 手动合并一次 — 可视化合并前后变化
5. 完整训练循环
6. 压缩比曲线 — vocab_size vs 序列长度
7. Encode/Decode 往返一致性测试
8. 词表分析 — 学到的中英文 token
9. 与 tiktoken (GPT-4) 对比

## 算法原理

**Byte-Level BPE** (GPT-2 做法):

```
文本 "你好" → UTF-8 字节 [228, 189, 160, 229, 165, 189]
                         ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
                            "你" 3字节        "好" 3字节
```

1. 初始词表 = 256 个字节 token (0x00-0xFF)
2. 统计所有相邻 pair 的频率
3. 将最高频 pair 合并为新 token（id = 256, 257, ...）
4. 重复直到达到目标词表大小

**编码时**按训练顺序依次应用合并规则，**解码时**拼接每个 token 对应的字节再 UTF-8 解码。

## API 参考

```python
class ByteLevelBPETokenizer:
    def train(self, text: str, vocab_size: int = 1256, verbose: bool = False)
    def encode(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str
    def save(self, path: str)
    def load(self, path: str)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `merges` | `list[tuple[tuple[int,int], int]]` | 合并规则，按训练顺序 |
| `vocab` | `dict[int, bytes]` | 词表，token_id → 字节 |

## 训练结果

在 1.3MB 中英混合语料上训练 1000 次合并：

- 词表大小：1256
- 压缩比：2.60x
- 包含中文 token：205 个
- 往返一致性：全部通过（空字符串、中文、英文、emoji）
