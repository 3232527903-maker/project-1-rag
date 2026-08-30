"""
Day 4 单测：验证「建表 -> 插入 -> 检索」链路。

关键设计：
- 用临时库文件（test_milvus_lite.db），不碰正式库 ./milvus_lite.db；
- 用手造假向量（固定值），不调用真实 Embedding API —— 测试不依赖网络、可离线稳定跑。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_vector_store.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ingest.chunker import Chunk
from ingest.vector_store import COLLECTION_NAME, _make_client, ensure_collection, insert_chunks, search

TEST_URI = "./test_milvus_lite.db"
DIM = 8  # 假向量的维度（测试用，小一点跑得快）


@pytest.fixture(scope="module")
def client():
    """每个测试模块一个临时 client：建表 -> yield -> 清理。"""
    c = _make_client(TEST_URI)
    if c.has_collection(COLLECTION_NAME):
        c.drop_collection(COLLECTION_NAME)
    ensure_collection(c, dim=DIM)
    yield c
    if c.has_collection(COLLECTION_NAME):
        c.drop_collection(COLLECTION_NAME)
    c.close()


def _make_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(
            text=f"测试文档第 {i} 段的内容",
            source="https://example.com/test",
            title="测试文档",
            chunk_index=i,
        )
        for i in range(n)
    ]


def _make_vecs(n: int = 3) -> list[list[float]]:
    """造 n 个假向量：第 i 个向量在第一个维度上更接近 0.9，保证能被搜到。"""
    return [[0.9 if j == 0 else 0.1 for j in range(DIM)] for _ in range(n)]


def test_collection_created(client):
    """建表后 has_collection 应为 True。"""
    assert client.has_collection(COLLECTION_NAME)


def test_insert_and_count(client):
    """插入后 row_count 应等于插入条数。"""
    insert_chunks(client, _make_chunks(3), _make_vecs(3))
    client.flush(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)
    assert stats["row_count"] == 3


def test_search_returns_topk(client):
    """检索应返回结果，且第一条命中携带元数据（溯源字段）。"""
    hits = search(client, _make_vecs(1)[0], top_k=3)
    assert len(hits) > 0
    first = hits[0]
    assert first["entity"]["title"] == "测试文档"
    assert "text" in first["entity"]
    assert "chunk_index" in first["entity"]
    # COSINE 度量下，和查询向量方向一致的向量 score 应明显高于 0
    assert first["distance"] > 0.5
