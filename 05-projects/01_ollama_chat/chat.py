"""
项目一: 本地 CLI 聊天助手
使用 Ollama API 实现多轮对话

功能:
- 多轮对话 (带历史记录)
- 流式输出
- 系统提示词支持

使用方法:
1. 确保 ollama 服务运行中: ollama serve
2. 运行: python chat.py
"""

import requests
import json
import sys


def chat_with_ollama(
    model: str = "llama3.2:3b",
    system_prompt: str = "You are a helpful assistant.",
):
    """
    与 Ollama 模型进行多轮对话

    Args:
        model: 模型名称
        system_prompt: 系统提示词
    """
    # Ollama API 地址
    api_url = "http://localhost:11434/api/chat"

    # 对话历史
    messages = []

    # 添加系统消息
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    print(f"🤖 使用模型: {model}")
    print(f"📝 系统提示: {system_prompt}")
    print("-" * 50)
    print("开始对话 (输入 'quit' 退出, 'clear' 清空历史)")
    print("-" * 50)

    while True:
        # 获取用户输入
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("再见!")
            break

        if user_input.lower() == 'clear':
            messages = messages[:1] if system_prompt else []
            print("历史已清空")
            continue

        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_input
        })

        # 构建请求
        payload = {
            "model": model,
            "messages": messages,
            "stream": True  # 启用流式输出
        }

        print("\n🤖 Assistant: ", end="", flush=True)

        try:
            # 发送请求 (流式)
            response = requests.post(api_url, json=payload, stream=True)
            response.raise_for_status()

            # 收集完整回复
            full_response = ""

            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        chunk = data["message"].get("content", "")
                        print(chunk, end="", flush=True)
                        full_response += chunk

                    # 检查是否完成
                    if data.get("done", False):
                        break

            print()  # 换行

            # 添加助手回复到历史
            messages.append({
                "role": "assistant",
                "content": full_response
            })

        except requests.exceptions.ConnectionError:
            print("\n❌ 无法连接到 Ollama 服务")
            print("   请确保运行了: ollama serve")
            # 移除未成功的用户消息
            messages.pop()
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            messages.pop()


def simple_completion(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    简单的单次补全 (非流式)

    Args:
        prompt: 输入提示
        model: 模型名称

    Returns:
        模型回复
    """
    api_url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(api_url, json=payload)
    response.raise_for_status()

    return response.json()["response"]


if __name__ == "__main__":
    # 可以通过命令行参数指定模型
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"

    # 自定义系统提示词
    system_prompt = """You are a helpful AI assistant.
You explain things clearly and concisely.
When discussing code, provide examples when helpful."""

    chat_with_ollama(model=model, system_prompt=system_prompt)
