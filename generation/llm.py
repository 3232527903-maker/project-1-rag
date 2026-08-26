"""
生成层（Day 5 实现）

职责：把检索到的上下文拼进 Prompt 模板，调用 DeepSeek 生成答案。

计划实现：
- 用 langchain-openai 的 ChatOpenAI（DeepSeek 走 OpenAI 兼容接口）
- Prompt 模板：角色设定 + 上下文 + 问题 + 「仅根据上下文回答，找不到就说不知道」约束
- 输出带引用溯源的答案
"""

# TODO(Day 5)：实现 LLM 调用与 Prompt 模板，例如：
# def generate(contexts: list[dict], question: str) -> str:
#     """基于检索上下文 + 问题，调用 LLM 生成答案。"""
#     raise NotImplementedError("Day 5 实现")
