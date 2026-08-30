"""
一键入库脚本（Day 4）

流程：load_all（84 篇）-> chunk_all（519 个）-> 分批 embed_texts -> insert_chunks -> 核对行数。

运行方式（项目根目录下）：
    .venv/bin/python ingest/build_index.py
"""

import sys
from pathlib import Path

# 直接运行时把项目根加进 sys.path（否则 import 不到 config / ingest 包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.chunker import chunk_all
from ingest.embedder import embed_texts, probe_dimension
from ingest.loader import load_all
from ingest.vector_store import COLLECTION_NAME, BATCH_SIZE, _make_client, ensure_collection, insert_chunks


def main() -> None:
    client = _make_client()

    # 1) 读取 + 分块（复用 Day 2 / Day 3 的成果）
    docs = load_all()
    chunks = chunk_all(docs)
    print(f"共 {len(docs)} 篇 -> {len(chunks)} 个 chunk")

    # 2) 探维度 -> 建表（dim 必须和模型输出一致）
    dim = probe_dimension()
    print(f"Embedding 维度: {dim}")
    ensure_collection(client, dim=dim)

    # 3) 分批向量化 + 插入（每批 20 条：DashScope embedding 单批上限 20，超过报 400；
    #    小批也便于单批失败只重试这一批）
    total = len(chunks)
    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vecs = embed_texts([c.text for c in batch])
        insert_chunks(client, batch, vecs)
        inserted += len(batch)
        print(f"进度: {inserted}/{total}")

    # 4) 核对入库行数（flush 确保落盘后可查）
    client.flush(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)
    row_count = stats.get("row_count")
    print(f"入库完成: collection 内行数 = {row_count}（应等于 {total}）")


if __name__ == "__main__":
    main()
