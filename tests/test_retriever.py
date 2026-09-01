"""
Day 6 单测：验证 retriever 的 retrieve 包装逻辑（monkeypatch 假数据，不依赖网络）。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_retriever.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import retriever


def test_retrieve_返回字段齐全(monkeypatch):
    """检索结果应包含 text/source/title/chunk_index/score 五个字段，score 四舍五入。"""
    fake_hits = [
        {
            "entity": {
                "text": "正文",
                "source": "http://s",
                "title": "T",
                "chunk_index": 3,
            },
            "distance": 0.67890,
        },
    ]
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1] * 4)
    monkeypatch.setattr(retriever, "_make_client", lambda: object())
    monkeypatch.setattr(retriever, "search", lambda c, v, top_k=4: fake_hits)

    rs = retriever.retrieve("问题", top_k=4)
    assert len(rs) == 1
    assert set(rs[0].keys()) == {"text", "source", "title", "chunk_index", "score"}
    assert rs[0]["chunk_index"] == 3
    assert rs[0]["score"] == 0.6789  # round(0.67890, 4)


def test_retrieve_top_k_透传(monkeypatch):
    """top_k 应透传给 search。"""
    calls = {}

    def fake_search(client, vec, top_k=4):
        calls["top_k"] = top_k
        return []

    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1] * 4)
    monkeypatch.setattr(retriever, "_make_client", lambda: object())
    monkeypatch.setattr(retriever, "search", fake_search)

    retriever.retrieve("问题", top_k=2)
    assert calls["top_k"] == 2


def test_retrieve_空结果返回空列表(monkeypatch):
    """无命中时返回空列表而非报错。"""
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1] * 4)
    monkeypatch.setattr(retriever, "_make_client", lambda: object())
    monkeypatch.setattr(retriever, "search", lambda c, v, top_k=4: [])

    assert retriever.retrieve("问题") == []
