"""
检索层（Day 5 实现）

职责：输入 query -> Embedding -> 在 Milvus 里做相似度检索 -> 返回 Top-K 相关片段。

计划实现（第 1 周先用单一向量检索，混合检索/重排留到第 2 周）：
- 用通义 text-embedding-v3 把 query 转成向量
- 在 Milvus Collection 里做向量相似度检索（余弦相似度）
- 返回 Top-K chunk 及其元数据
"""

# TODO(Day 5)：实现向量检索函数，例如：
# def retrieve(query: str, top_k: int = 4) -> list[dict]:
#     """检索与 query 最相关的 top_k 个片段，返回 [{text, metadata, score}, ...]。"""
#     raise NotImplementedError("Day 5 实现")
