"""
端到端问答入口（Day 5 里程碑）：提问 -> 检索 -> 生成 -> 带引用的答案。

运行方式（项目根目录下）：
    .venv/bin/python qa.py "你的问题"
"""

from generation.llm import generate
from retrieval.retriever import retrieve


def ask(question: str, top_k: int = 4) -> None:
    print(f"Q: {question}")
    contexts = retrieve(question, top_k=top_k)
    if not contexts:
        print("未检索到相关内容，请检查向量库是否已入库。")
        return

    print(f"\n--- 检索到 {len(contexts)} 条上下文 ---")
    for c in contexts:
        print(f"  [{c['title']}] 第{c['chunk_index']}段 score={c['score']}")

    answer = generate(contexts, question)
    print(f"\nA: {answer}")

    print("\n--- 引用来源 ---")
    for c in contexts:
        print(f"  {c['title']} : {c['source']}")


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "如何配置模型 API Key？"
    ask(q)