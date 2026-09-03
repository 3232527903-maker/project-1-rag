"""
Day 8 单测：验证 RRF 融合排序（rrf_fuse 是纯函数，用假的两路 hits 即可，不依赖真实检索器）。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_hybrid.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid import rrf_fuse

# 假数据：同一 chunk 用 source + "|" + chunk_index 唯一标识
VEC_A0 = {"source": "a.md", "chunk_index": 0, "title": "A", "text": "a 的第 0 块"}  # 向量侧第 1 名
VEC_A1 = {"source": "a.md", "chunk_index": 1, "title": "A", "text": "a 的第 1 块"}  # 向量侧第 2 名
BM_B0 = {"source": "b.md", "chunk_index": 0, "title": "B", "text": "b 的第 0 块"}  # BM25 侧第 1 名
BM_A0 = {"source": "a.md", "chunk_index": 0, "title": "A", "text": "a 的第 0 块"}  # BM25 侧第 2 名（与 VEC_A0 同一 chunk）


def test_rrf_两路都出现的chunk应排第一():
    """a.md|0 在两路分别排第 1 / 第 4 名，RRF 分累加后应超越只出现一次的其他 chunk。"""
    result = rrf_fuse(vector_hits=[VEC_A0, VEC_A1], bm25_hits=[BM_B0, BM_A0], top_k=3)
    assert result[0]["source"] == "a.md"
    assert result[0]["chunk_index"] == 0


def test_rrf_融合后排序正确():
    """期望顺序：a.md|0（双路都靠前）> a.md|1（向量第 2）> b.md|0（BM25 第 1）。"""
    result = rrf_fuse(vector_hits=[VEC_A0, VEC_A1], bm25_hits=[BM_B0, BM_A0], top_k=3)
    keys = [(h["source"], h["chunk_index"]) for h in result]
    assert keys == [("a.md", 0), ("a.md", 1), ("b.md", 0)]


def test_rrf_top_k_截断():
    """top_k 应生效：返回条数不超过 top_k。"""
    result = rrf_fuse(vector_hits=[VEC_A0, VEC_A1], bm25_hits=[BM_B0, BM_A0], top_k=2)
    assert len(result) == 2
