"""
向量存储层（Day 4 实现）

职责：封装 Milvus 的「建表 -> 建索引 -> 插入 -> 检索」。
- 用 pymilvus 3.x 推荐的 MilvusClient，uri 指向本地文件即自动进入 Milvus Lite 模式；
- Collection 固定字段：id / text / source / title / chunk_index / vector；
- 度量用 COSINE（文本语义相似度，看方向不看长度）。

为什么是这五个字段：
- vector：检索匹配用；
- text：命中后要取回原文，才能拼进 Prompt 喂给 LLM；
- source / title：溯源（面试点：RAG 答案要能回到原文出处）；
- chunk_index：命中后知道是原文第几段。

使用方式：
    from ingest.vector_store import ensure_collection, insert_chunks, search
"""

import sys
from pathlib import Path

# 直接运行时（python ingest/vector_store.py）项目根不在 sys.path，
# 把它加进去，否则 import 不到 config 包。
# 作为包被 import 时（from ingest.vector_store import ...）__package__ 非空，跳过。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import DataType, MilvusClient

from config.settings import settings

COLLECTION_NAME = "rag_docs"
TEXT_MAX_LENGTH = 1024   # 512 字 chunk 留 2 倍余量
BATCH_SIZE = 20          # 批量向量化的每批条数：DashScope embedding 单批上限 20（超过报 400）


def _make_client(uri: str | None = None) -> MilvusClient:
    """创建 MilvusClient。uri 指向本地文件（如 ./milvus_lite.db），自动用 Lite 模式。

    传入 uri 参数是为了测试时用临时库，不污染正式库。

    坑：.env 里自定义键名是 MILVUS_LOCAL_DB（settings.MILVUS_LOCAL_DB），
    不能用 MILVUS_URI——pymilvus 3.x import 时会自动加载 .env，
    读到本地路径会当远程地址解析而崩溃。
    """
    return MilvusClient(uri=uri or settings.MILVUS_LOCAL_DB)


def ensure_collection(client: MilvusClient, dim: int, collection_name: str = COLLECTION_NAME) -> None:
    """确保 Collection 存在（不存在才创建）。

    建表五步：定字段(schema) -> 建索引(index) -> create_collection。
    dim 必须等于 Embedding 模型实际输出维度（先 probe_dimension 再调用本函数）。
    """
    if client.has_collection(collection_name):
        print(f"Collection {collection_name} 已存在，跳过创建")
        return

    # 1) 定义字段：auto_id=True 让主键自动生成，不用自己管
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=TEXT_MAX_LENGTH)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=TEXT_MAX_LENGTH)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    # 2) 建索引：数据量小（519 条）用 FLAT 全量精确比对即可，第 2 周数据量大了再换 HNSW
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")

    # 3) 建表
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection {collection_name} 创建完成（dim={dim}, metric=COSINE）")


def insert_chunks(client: MilvusClient, chunks: list, vecs: list[list[float]],
                  collection_name: str = COLLECTION_NAME) -> None:
    """把 Chunk 列表 + 对应向量批量插入。

    注意：chunks 与 vecs 按下标一一对应（来自同一次 embed_texts），
    组装成 Milvus 需要的 [{字段: 值, ...}] 结构。
    """
    data = [
        {
            "text": c.text,
            "source": c.source,
            "title": c.title,
            "chunk_index": c.chunk_index,
            "vector": v,
        }
        for c, v in zip(chunks, vecs)
    ]
    res = client.insert(collection_name=collection_name, data=data)
    print(f"本次插入 {len(data)} 条")


def search(client: MilvusClient, query_vec: list[float], top_k: int = 5,
           collection_name: str = COLLECTION_NAME) -> list[dict]:
    """向量检索，返回 top_k 个命中。

    返回的每个元素形如：
        {"id": ..., "distance": 0.87, "entity": {"text": ..., "source": ..., "title": ..., "chunk_index": 3}}
    distance 在 COSINE 度量下越接近 1 越相关。

    Milvus 坑：search/query 前必须先 load_collection()（把集合加载进内存），
    否则报 code=101 "Collection is in state 'released'"。load 是幂等的，重复调用无害。
    """
    client.load_collection(collection_name)
    res = client.search(
        collection_name=collection_name,
        data=[query_vec],
        limit=top_k,
        output_fields=["text", "source", "title", "chunk_index"],
    )
    return res[0]   # 只搜了一个 query，取第一个（也是唯一一个）query 的结果列表
