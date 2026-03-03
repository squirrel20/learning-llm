# LLM 本地学习项目

本地大语言模型 (LLM) 学习环境，包含从基础原理到推理优化的完整学习路径。

## 环境状态

| 组件 | 状态 |
|------|------|
| Ollama | ✅ 已安装，llama3.2:3b 已下载 |
| llama.cpp | ✅ 已编译 (Metal 加速) |

## 目录结构

```
learning-llm/
├── 01-basics/           # 基础原理 (Tokenization, etc.)
├── 02-transformer/      # Transformer 架构实现
├── 03-llama-cpp/        # llama.cpp 源码与学习笔记
├── 04-optimization/     # 推理优化实验
├── 05-projects/         # 实践项目
├── models/              # 本地 GGUF 模型文件
└── notes/               # 学习笔记与检查清单
```

## 快速开始

### 1. 测试 Ollama
```bash
# 启动服务 (如果未运行)
ollama serve

# 交互式对话
ollama run llama3.2:3b
```

### 2. 测试 llama.cpp
```bash
# 需要先下载 GGUF 模型到 models/ 目录
./03-llama-cpp/llama.cpp/build/bin/llama-cli -m models/your-model.gguf -p "Hello"
```

### 3. 运行学习示例
```bash
# Tokenization 演示
python 01-basics/tokenization_demo.py

# Self-Attention 演示
python 02-transformer/simple_attention.py

# 聊天助手项目
python 05-projects/01_ollama_chat/chat.py
```

## 学习路径

详见 `notes/learning_checklist.md`

## 常用命令

```bash
# Ollama
ollama list                    # 查看已下载模型
ollama pull <model>           # 下载模型
ollama run <model>            # 运行模型

# llama.cpp
./llama-cli -m <model> -p <prompt>    # 推理
./llama-bench -m <model>              # 性能测试
./llama-quantize <in> <out> <type>    # 量化模型
```

## 推荐学习资源

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy
- [llama.cpp Wiki](https://github.com/ggerganov/llama.cpp/wiki)
