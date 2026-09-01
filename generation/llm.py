"""
生成层（Day 5 实现）

职责：把检索到的上下文拼进 Prompt 模板，调用 DeepSeek 生成答案。
- 用 openai SDK 直调 DeepSeek（OpenAI 兼容接口），不用 langchain 封装
  （langchain-openai 1.6 的请求体格式兼容接口不认，实测报 400）
- Prompt 模板：角色设定 + 上下文 + 问题 + 「仅根据资料回答，找不到就说不知道」约束
- temperature=0.3：偏低，减少自由发挥（防幻觉三板斧之一）
"""

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

from config.settings import settings

_client = OpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
)

SYSTEM_PROMPT = (
    "你是一个严谨的智能问答助手。"
    "你只能依据用户提供的资料回答问题，禁止编造任何内容。"
    "如果资料中没有相关内容，请明确回答：资料中没有相关内容。"
)
def _build_context(contexts: list[dict]) -> str:
    """把检索片段拼成带编号、带来源的文本块。"""
    parts = []
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[{i}] 来源：{ctx['title']}（{ctx['source']}）\n{ctx['text']}")
    return "\n\n".join(parts)

def generate(
    contexts: list[dict],
    question: str,
    temperature: float = 0.3,
    max_retries: int = 2,
) -> str:
    """基于检索上下文 + 问题，调用 LLM 生成答案。

    容错分级（Day 6 新增，面试考点）：
    - 可重试（网络类）：超时 / 连接失败 / 限流(429) -> 指数退避重试
    - 不可重试（业务类）：401 鉴权失败、400 参数错误 -> 重试也没用，直接抛出
    """
    context_text = _build_context(contexts)
    # 注意：user_prompt 必须是「字符串」！之前误写成 [f"..."]（列表），
    # content 会被序列化成裸字符串数组 ["..."],DeepSeek 报
    # invalid type: string ... expected ChatCompletionRequestContentBlock(400)。
    user_prompt = (
        f"以下是相关资料：\n{context_text}\n\n"
        f"请仅依据上述资料回答问题：{question}"
    )
    # 只重试「网络类」错误：超时 / 连接失败 / 限流
    retryable = (APITimeoutError, APIConnectionError, RateLimitError)
    for attempt in range(max_retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except retryable:
            if attempt == max_retries:
                raise  # 重试耗尽，交给上层兜底
            time.sleep(1 + attempt)  # 退避：等 1s、2s
if __name__ == '__main__':
    import json
    from retrieval.retriever import retrieve
    question = "如何配置模型 API Key？"
    contexts = retrieve(question, top_k=3)  # 检索与生成用同一问题，避免演示时防幻觉误触发
    print(generate(contexts=contexts, question=question))