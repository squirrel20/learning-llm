"""
Tokenization 基础演示
理解 LLM 如何将文本转换为数字

学习目标:
1. 理解为什么需要 Tokenization
2. 了解不同的 Tokenization 方法
3. 动手使用 tiktoken (OpenAI) 和 sentencepiece
"""

def basic_tokenization():
    """基础分词概念演示 (不需要外部库)"""
    print("=" * 60)
    print("基础分词概念")
    print("=" * 60)

    text = "Hello, world! 你好世界"

    # 方法1: 字符级分词
    char_tokens = list(text)
    print(f"\n原文: {text}")
    print(f"\n1. 字符级分词:")
    print(f"   tokens: {char_tokens}")
    print(f"   词表大小问题: 需要包含所有可能的字符")

    # 方法2: 单词级分词
    word_tokens = text.split()
    print(f"\n2. 单词级分词:")
    print(f"   tokens: {word_tokens}")
    print(f"   问题: 词表会非常大, 无法处理新词")

    # 方法3: 子词分词 (BPE/SentencePiece 的思想)
    print(f"\n3. 子词分词 (Subword Tokenization):")
    print("""
   核心思想: 在字符和单词之间找平衡

   常见算法:
   - BPE (Byte-Pair Encoding): GPT 系列使用
   - WordPiece: BERT 使用
   - SentencePiece: LLaMA 使用

   优点:
   - 词表大小可控 (通常 32k-100k)
   - 可以处理任何输入 (包括新词)
   - 常见词用一个 token, 罕见词拆分成多个
""")


def bpe_demo():
    """BPE (Byte-Pair Encoding) 算法演示"""
    print("\n" + "=" * 60)
    print("BPE 算法简化演示")
    print("=" * 60)

    # 简化的 BPE 演示
    # 实际 BPE 会在整个语料库上学习合并规则

    corpus = ["low", "lower", "newest", "widest"]
    print(f"\n训练语料: {corpus}")

    print("""
BPE 训练过程 (简化):

初始状态: 每个字符是一个 token
['l', 'o', 'w', '</w>']
['l', 'o', 'w', 'e', 'r', '</w>']
['n', 'e', 'w', 'e', 's', 't', '</w>']
['w', 'i', 'd', 'e', 's', 't', '</w>']

Step 1: 找最常见的相邻 token 对 -> ('e', 's') 出现 2 次
        合并为 'es'

Step 2: 找下一个最常见的对 -> ('es', 't') 出现 2 次
        合并为 'est'

Step 3: 继续直到达到目标词表大小...

最终词表可能包含: ['l', 'o', 'w', 'e', 'r', 'n', 'i', 'd', 's', 't',
                  'lo', 'low', 'er', 'est', 'new', ...]
""")


def tiktoken_demo():
    """使用 tiktoken (OpenAI 的分词器) 的演示"""
    print("\n" + "=" * 60)
    print("tiktoken 演示 (需要安装: pip install tiktoken)")
    print("=" * 60)

    try:
        import tiktoken

        # GPT-4 使用的分词器
        enc = tiktoken.get_encoding("cl100k_base")

        texts = [
            "Hello, world!",
            "你好世界",
            "The quick brown fox jumps over the lazy dog.",
            "LLaMA is a large language model.",
        ]

        for text in texts:
            tokens = enc.encode(text)
            decoded = enc.decode(tokens)

            print(f"\n原文: {text}")
            print(f"Token IDs: {tokens}")
            print(f"Token 数量: {len(tokens)}")

            # 显示每个 token 对应的文本
            token_texts = [enc.decode([t]) for t in tokens]
            print(f"Token 文本: {token_texts}")

    except ImportError:
        print("\ntiktoken 未安装")
        print("安装命令: pip install tiktoken")
        print("\n示例输出 (使用 cl100k_base):")
        print("""
原文: Hello, world!
Token IDs: [9906, 11, 1917, 0]
Token 数量: 4
Token 文本: ['Hello', ',', ' world', '!']

原文: 你好世界
Token IDs: [57668, 53901]
Token 数量: 2
Token 文本: ['你好', '世界']
""")


def sentencepiece_demo():
    """SentencePiece (LLaMA 使用) 的概念说明"""
    print("\n" + "=" * 60)
    print("SentencePiece (LLaMA 使用的分词器)")
    print("=" * 60)

    print("""
SentencePiece 特点:

1. 语言无关: 直接在原始文本上训练, 不需要预分词
2. 子词单元: 同样使用 BPE 或 Unigram 算法
3. 特殊处理: 用 ▁ 表示空格 (所以 "Hello world" 变成 ["▁Hello", "▁world"])

LLaMA 分词器配置:
- 词表大小: 32000
- 算法: BPE
- 特殊 token: <s> (BOS), </s> (EOS), <unk> (Unknown)

使用示例 (需要下载 LLaMA tokenizer):

```python
from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode("Hello, world!")
print(tokens)  # [15043, 29892, 3186, 29991]
```

安装: pip install sentencepiece
""")


def token_embedding_concept():
    """Token Embedding 概念说明"""
    print("\n" + "=" * 60)
    print("从 Token 到向量: Embedding")
    print("=" * 60)

    print("""
Tokenization 之后的流程:

1. 文本 -> Token IDs
   "Hello" -> [15043]

2. Token IDs -> Embedding 向量
   [15043] -> [0.1, -0.3, 0.5, ..., 0.2]  (例如 4096 维)

3. Embedding 矩阵
   - 大小: vocab_size × embedding_dim
   - 例如: 32000 × 4096 (LLaMA)
   - 每行是一个 token 的向量表示
   - 这些向量在训练中学习得到

4. 为什么需要 Embedding?
   - 将离散的 token 映射到连续的向量空间
   - 语义相似的词在向量空间中距离更近
   - 神经网络只能处理数值输入
""")


if __name__ == "__main__":
    basic_tokenization()
    bpe_demo()
    tiktoken_demo()
    sentencepiece_demo()
    token_embedding_concept()

    print("\n" + "=" * 60)
    print("练习建议")
    print("=" * 60)
    print("""
1. 安装并使用 tiktoken:
   pip install tiktoken

2. 比较不同文本的 token 数量:
   - 英文 vs 中文
   - 常见词 vs 专业术语
   - 代码 vs 自然语言

3. 阅读 OpenAI 的 tokenizer 工具:
   https://platform.openai.com/tokenizer
""")
