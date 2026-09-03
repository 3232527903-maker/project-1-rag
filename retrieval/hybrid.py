"""
检索融合层（Day 8 实现）

职责：输入向量检索 + BM25 检索两路 Top-K -> RRF（倒数排名融合）-> 返回融合后的 Top-K 片段。

实现要点（混合检索第二步；Day 8 先单独验证效果，第 2 周再接入 qa.py）：
- RRF 公式：score(chunk) = Σ 1 / (k + rank)，k 通常取 60
- 只按名次累加、不看原始分数 → 绕开 BM25 分与余弦相似度量纲不可比的问题
- 同一 chunk 用 source + "|" + chunk_index 作唯一标识，两路榜单都出现时分数累加
- 含义：两边都靠前的 chunk 比只在单边靠前的更可信
"""

# TODO(Day 8)：实现 RRF 融合（手敲练习）与混合检索包装，例如：
# def rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60,
#              top_k: int = 5) -> list[dict]:
#     """倒数排名融合：同一条记录在两路名次越高，融合分越高。"""
#     ...
#
# def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
#     """向量检索 + BM25 检索，走 rrf_fuse 后返回 Top-K。"""
#     ...
def rrf_fuse(vector_hits:list[dict],bm25_hits:list[dict],k:int = 60,top_k:int = 5)->list[dict]:
    """倒数排名融合：同一条记录在两路名次越高，融合分越高。"""
    scores :dict[str,float]= {} #用来计算分数
    items :dict[str,dict]= {} #用来记录返回的数据的格式
    for rank,hit in enumerate(vector_hits+bm25_hits,1):
        key = hit["source"] +"|" + str(hit["chunk_index"])
        scores[key] = scores.get(key,0.0) +1.0/(k+rank)
        items[key] = hit
        ranked = sorted(scores.items(),key=lambda kv:kv[1],reverse=True)
    return [items[key] for key,_ in ranked[:top_k]]