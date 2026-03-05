"""
下载训练语料
来源: Project Gutenberg (Tiny Shakespeare)
"""

import urllib.request
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")


def download_text(url: str) -> str:
    """下载文本文件"""
    print(f"下载: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def prepare_corpus():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 英文: Tiny Shakespeare (~1MB)
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = download_text(shakespeare_url)
    print(f"  英文语料: {len(text):,} 字符")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    size_mb = os.path.getsize(CORPUS_PATH) / (1024 * 1024)
    print(f"\n语料已保存到: {CORPUS_PATH}")
    print(f"总大小: {size_mb:.2f} MB")
    print(f"总字符数: {len(text):,}")


if __name__ == "__main__":
    prepare_corpus()
