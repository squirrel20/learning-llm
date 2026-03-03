# LLM 学习路径检查清单

## 阶段一: 环境搭建与快速体验 ✅

- [x] 安装 Ollama
- [x] 下载 llama3.2:3b 模型
- [x] 测试 Ollama 推理
- [x] 克隆 llama.cpp
- [x] 编译 llama.cpp (Metal 加速)
- [ ] 下载 GGUF 模型测试 llama.cpp

## 阶段二: LLM 核心原理

### 神经网络基础
- [ ] 理解前向传播
- [ ] 理解反向传播
- [ ] 理解损失函数
- [ ] 观看 3Blue1Brown 神经网络系列

### Transformer 架构
- [ ] 阅读 "The Illustrated Transformer"
- [ ] 理解 Self-Attention 机制
- [ ] 理解 Multi-Head Attention
- [ ] 理解 Position Encoding
- [ ] 理解 Layer Normalization
- [ ] 理解 Feed-Forward Network
- [ ] 运行 `02-transformer/simple_attention.py`

### GPT 架构
- [ ] 理解 Decoder-only 架构
- [ ] 理解 Causal Attention
- [ ] 理解 Next Token Prediction
- [ ] 理解 Tokenization (BPE)
- [ ] 运行 `01-basics/tokenization_demo.py`

### 推荐阅读/观看
- [ ] 论文: "Attention Is All You Need"
- [ ] 视频: Andrej Karpathy "Let's build GPT"
- [ ] 代码: 阅读 nanoGPT

## 阶段三: llama.cpp 深入理解

### 代码结构
- [ ] 阅读 `include/llama.h` - API 设计
- [ ] 阅读 `tools/main/main.cpp` - 推理入口
- [ ] 阅读 `src/llama.cpp` - 核心推理
- [ ] 阅读 `ggml/src/ggml.c` - 张量运算
- [ ] 阅读 `ggml/src/ggml-metal.m` - Metal 加速

### 关键概念
- [ ] 理解 GGUF 文件格式
- [ ] 理解 ggml 张量库
- [ ] 理解 KV Cache 机制
- [ ] 理解 Metal GPU 加速原理

## 阶段四: 推理优化技术

### 量化
- [ ] 理解 INT8/INT4 量化原理
- [ ] 理解 GGUF 量化级别
- [ ] 实践: 使用 llama-quantize 量化模型
- [ ] 对比不同量化级别的精度/速度

### KV Cache 优化
- [ ] 理解 Flash Attention
- [ ] 理解 PagedAttention
- [ ] 理解 KV Cache 量化

### 性能分析
- [ ] 使用 llama-bench 测试
- [ ] 分析 tokens/second
- [ ] 分析内存占用

## 阶段五: 实践项目

### 项目一: 本地 CLI 聊天助手
- [x] 基础框架已创建
- [ ] 添加更多功能
- [ ] 测试不同模型

### 项目二: 简单 RAG 系统
- [ ] 实现文档切分
- [ ] 集成向量数据库
- [ ] 实现检索增强生成

### 项目三: 模型性能对比
- [ ] 对比不同量化级别
- [ ] 速度 vs 质量分析
- [ ] 可视化结果

---

## 学习资源链接

### 视频
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

### 文章
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

### 代码仓库
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [llm.c](https://github.com/karpathy/llm.c)

### 社区
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- [Hugging Face](https://huggingface.co/)
- [llama.cpp Discussions](https://github.com/ggerganov/llama.cpp/discussions)
