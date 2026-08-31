"""
生成层（Day 5 实现）

职责：把检索到的上下文拼进 Prompt 模板，调用 DeepSeek 生成答案。
- 用 openai SDK 直调 DeepSeek（OpenAI 兼容接口），不用 langchain 封装
  （langchain-openai 1.6 的请求体格式兼容接口不认，实测报 400）
- Prompt 模板：角色设定 + 上下文 + 问题 + 「仅根据资料回答，找不到就说不知道」约束
- temperature=0.3：偏低，减少自由发挥（防幻觉三板斧之一）
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

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

def generate(contexts: list[dict], question: str, temperature: float = 0.3) -> str:
    """基于检索上下文 + 问题，调用 LLM 生成答案。"""
    context_text = _build_context(contexts)
    # 注意：user_prompt 必须是「字符串」！之前误写成 [f"..."]（列表），
    # content 会被序列化成裸字符串数组 ["..."],DeepSeek 报
    # invalid type: string ... expected ChatCompletionRequestContentBlock(400)。
    user_prompt = (
        f"以下是相关资料：\n{context_text}\n\n"
        f"请仅依据上述资料回答问题：{question}"
    )
    resp = _client.chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content
if __name__ == '__main__':
    import json
    from retrieval.retriever import retrieve
    contexts = retrieve("如何配置模型 API Key？", top_k=3)  # 直接传 list[dict]
    print(generate(contexts=contexts, question="如何调用大模型 API？"))