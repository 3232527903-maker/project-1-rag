"""
检索验证脚本（Day 4）

职责：验证「向量库能检索」——query -> Embedding -> Milvus 相似度搜索 -> 打印 Top-K（带溯源）。
注意：这是 Day 4 的验证脚本；正式检索层 retrieval/retriever.py 留到 Day 5 实现。

运行方式（项目根目录下）：
    .venv/bin/python retrieval/search.py
"""

import sys
from pathlib import Path

# 直接运行时把项目根加进 sys.path（retrieval/ 下 import 不到 config / ingest）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.embedder import embed_query
from ingest.vector_store import COLLECTION_NAME, _make_client, search


def show_result(question: str, top_k: int = 3) -> None:
    client = _make_client()
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} 不存在，请先跑 ingest/build_index.py 入库")
        return

    vec = embed_query(question)
    hits = search(client, vec, top_k=top_k)
    print("=" * 64)
    print(f"Q: {question}")
    if not hits:
        print("  没有命中结果")
        return
    for h in hits:
        entity = h.get("entity", {})
        text = entity.get("text", "")
        print(f"  score={h.get('distance', 0):.4f} | [{entity.get('title')}] 第{entity.get('chunk_index')}段")
        print(f"    {text[:60]}...")


def main() -> None:
    # 换成 2-3 个跟你的文档内容相关的问题来测
    questions = [
        "什么是 Milvus？",
        "什么是 RAG？",
        "如何调用大模型 API？",
    ]
    for q in questions:
        show_result(q)


if __name__ == "__main__":
    main()
