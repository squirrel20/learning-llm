#!/bin/bash
# llama.cpp 性能测试脚本
#
# 使用方法:
# 1. 下载一个 GGUF 模型到 models/ 目录
# 2. 运行: ./benchmark_test.sh models/your-model.gguf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="$SCRIPT_DIR/../03-llama-cpp/llama.cpp"
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <model.gguf>"
    echo ""
    echo "示例:"
    echo "  $0 ../models/llama-3.2-3b-q4_k_m.gguf"
    exit 1
fi

MODEL_PATH="$1"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查 llama-bench
if [ ! -f "$LLAMA_BENCH" ]; then
    echo "❌ llama-bench 不存在"
    echo "   请先编译 llama.cpp"
    exit 1
fi

echo "=================================================="
echo "llama.cpp 性能测试"
echo "=================================================="
echo ""
echo "模型: $MODEL_PATH"
echo "时间: $(date)"
echo ""

# 运行基准测试
echo "▶ 运行 llama-bench..."
echo ""

$LLAMA_BENCH -m "$MODEL_PATH" -p 128 -n 128 -r 3

echo ""
echo "=================================================="
echo "指标说明:"
echo "=================================================="
echo "
pp (prompt processing): 处理提示词的速度 (tokens/s)
tg (text generation):   生成文本的速度 (tokens/s)

影响因素:
- 模型大小: 更大的模型更慢
- 量化级别: Q4 比 Q8 更快,但质量略低
- Metal GPU: Apple Silicon 利用 GPU 加速
- 内存带宽: 主要瓶颈

优化建议:
1. 使用 Q4_K_M 量化在速度和质量间取得平衡
2. 确保 Metal 正确启用 (编译时自动检测)
3. 关闭其他占用 GPU 的应用
"

echo ""
echo "完成!"
