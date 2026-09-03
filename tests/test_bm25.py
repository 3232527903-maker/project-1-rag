"""
Day 8 单测：验证 BM25 中文分词检索（用几条假 chunk 构建索引即可，不依赖网络 / 向量库 / 真实文档）。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_bm25.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25_index import BM25Index


# 假数据：与入库结构一致，含 text / source / title / chunk_index 四个字段
FAKE_DOCS = [
    {
        "text": "FastAPI 是一个用于构建 API 的现代 Python Web 框架，支持类型校验与自动文档。",
        "source": "fastapi.md",
        "title": "FastAPI 入门",
        "chunk_index": 0,
    },
    {
        "text": "中文检索前必须先做分词，再把词项交给 BM25 计算相关性分数。",
        "source": "bm25.md",
        "title": "BM25 检索",
        "chunk_index": 0,
    },
    {
        "text": "Milvus 是一个开源的向量数据库，专门用来存向量并做相似度检索。",
        "source": "milvus.md",
        "title": "Milvus 向量库",
        "chunk_index": 1,
    },
]


def _build_index() -> BM25Index:
    return BM25Index(docs=FAKE_DOCS)


def test_search_中文关键词命中且排最前():
    """query 含中文关键词时，含该词的 chunk 应排在第 1 位（验证分词检索生效）。"""
    hits = _build_index().search("分词", top_k=3)
    assert hits[0]["source"] == "bm25.md"


def test_search_返回完整元数据与bm25_score():
    """返回条目应带 bm25_score，且 source / title / chunk_index 完整保留。"""
    hits = _build_index().search("Milvus", top_k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit["source"] == "milvus.md"
    assert hit["title"] == "Milvus 向量库"
    assert hit["chunk_index"] == 1
    assert isinstance(hit["bm25_score"], float)
    assert hit["bm25_score"] > 0.0


def test_search_top_k_截断():
    """top_k 应生效：返回条数不超过 top_k。"""
    hits = _build_index().search("检索", top_k=2)
    assert 0 < len(hits) <= 2


def test_search_无匹配词不报错():
    """没有任何词命中时不应抛异常，正常返回列表。"""
    hits = _build_index().search("完全不存在词xyzabc", top_k=3)
    assert isinstance(hits, list)
    assert len(hits) <= 3
