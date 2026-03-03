# llama.cpp 源码阅读指南

## 项目结构

```
llama.cpp/
├── src/
│   ├── llama.cpp           # 核心推理代码 (~15000行)
│   ├── llama-vocab.cpp     # 词表处理
│   └── llama-sampling.cpp  # 采样策略
├── ggml/
│   ├── src/
│   │   ├── ggml.c          # 张量运算核心
│   │   ├── ggml-metal.m    # Metal GPU 加速
│   │   ├── ggml-cuda.cu    # CUDA GPU 加速
│   │   └── ggml-backend.c  # 后端抽象层
│   └── include/
│       └── ggml.h          # ggml API 头文件
├── include/
│   └── llama.h             # 主要 API 头文件
├── tools/
│   ├── main/               # CLI 主程序 (llama-cli)
│   ├── server/             # HTTP 服务器
│   └── quantize/           # 量化工具
└── common/                 # 通用工具函数
```

## 推荐阅读顺序

### 第一阶段: 理解 API 设计

**文件: `include/llama.h`**

关键结构体:
- `llama_model` - 模型对象
- `llama_context` - 推理上下文
- `llama_batch` - 输入批次

关键函数:
```c
// 模型加载
llama_model * llama_model_load_from_file(const char * path_model, ...);

// 创建上下文
llama_context * llama_init_from_model(llama_model * model, ...);

// 推理
int llama_decode(llama_context * ctx, llama_batch batch);

// 采样
llama_token llama_sampler_sample(llama_sampler * smpl, llama_context * ctx, int idx);
```

### 第二阶段: 理解推理流程

**文件: `tools/main/main.cpp` (llama-cli 入口)**

阅读顺序:
1. `main()` 函数 - 整体流程
2. 参数解析
3. 模型加载
4. 推理循环

核心推理循环:
```cpp
while (还有token要生成) {
    // 1. 准备输入 batch
    llama_batch_add(batch, token, pos, seq_id, true);

    // 2. 执行推理 (前向传播)
    llama_decode(ctx, batch);

    // 3. 获取 logits (每个 vocab token 的分数)
    float * logits = llama_get_logits(ctx);

    // 4. 采样下一个 token
    llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

    // 5. 输出 token
    printf("%s", llama_token_to_piece(model, new_token));
}
```

### 第三阶段: 深入核心实现

**文件: `src/llama.cpp`**

关键概念:
1. **模型加载** (`llama_model_load`)
   - 读取 GGUF 文件头
   - 加载模型参数
   - 初始化张量

2. **KV Cache** (`llama_kv_cache`)
   - 存储历史 key/value
   - 避免重复计算

3. **前向传播** (`llama_build_graph` / `llama_decode_internal`)
   - 构建计算图
   - 执行矩阵运算

### 第四阶段: 底层张量运算

**文件: `ggml/src/ggml.c`**

关键数据结构:
```c
struct ggml_tensor {
    enum ggml_type type;    // 数据类型 (F32, F16, Q4_0, ...)
    int64_t ne[4];          // 每个维度的大小
    size_t  nb[4];          // 每个维度的步长 (stride)
    void * data;            // 数据指针
    // ...
};
```

关键运算:
```c
// 矩阵乘法
ggml_mul_mat(ctx, a, b);

// 层归一化
ggml_norm(ctx, x);

// RoPE 位置编码
ggml_rope(ctx, x, pos);

// Softmax
ggml_soft_max(ctx, x);
```

### 第五阶段: Metal GPU 加速

**文件: `ggml/src/ggml-metal.m`**

理解:
1. Metal kernel 如何调用
2. 内存管理 (CPU <-> GPU)
3. 关键操作的 GPU 实现

## 调试技巧

### 启用详细日志
```bash
# 设置环境变量
export LLAMA_METAL_NDEBUG=0

# 运行时查看 Metal 信息
./llama-cli -m model.gguf -p "test" --verbose
```

### 使用 llama-bench 测试性能
```bash
./llama-bench -m model.gguf -p 128 -n 128
```

### 查看模型信息
```bash
./llama-gguf model.gguf
```

## 关键概念速查

| 概念 | 说明 | 相关文件 |
|------|------|----------|
| GGUF | 模型文件格式 | `ggml/src/gguf.cpp` |
| ggml_tensor | 张量数据结构 | `ggml/include/ggml.h` |
| llama_batch | 输入批次 | `include/llama.h` |
| KV Cache | 键值缓存 | `src/llama.cpp` |
| 量化 | INT4/INT8 压缩 | `ggml/src/ggml-quants.c` |
| Metal | Apple GPU 加速 | `ggml/src/ggml-metal.m` |

## 学习资源

- [GGML 官方文档](https://github.com/ggerganov/ggml)
- [llama.cpp Wiki](https://github.com/ggerganov/llama.cpp/wiki)
- [Georgi Gerganov 的技术博客](https://ggerganov.com/)
