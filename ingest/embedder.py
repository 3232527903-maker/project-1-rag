"""
Embedding 层（Day 4 实现）

职责：把文本转成向量。
- 用 openai SDK 直连通义千问 DashScope 的 OpenAI 兼容接口（compatible-mode）；
- 封装三个函数：批量 embed、单条 query embed、探维度 probe_dimension。

为什么不用 langchain-openai 的 OpenAIEmbeddings：
- langchain-openai 1.6 发送的请求体格式 DashScope 兼容接口不认，
  实测报 400 `contents is neither str nor list of str`；
- 用 openai SDK 直调（input 传 list[str]）实测可通，少一层兼容问题。
（这是个真实的坑，面试可讲：兼容接口 ≠ 完全兼容，SDK 版本差异要实测。）

为什么封装成独立模块：
- 换 Embedding 供应商（第 2 周换本地 BGE）时，只改这一个文件，上层不用动；
- 面试可答「Embedding 抽象层：统一接口 + 可替换实现」。

使用方式：
    from ingest.embedder import embed_texts, embed_query, probe_dimension
"""

import sys
from pathlib import Path

# 直接运行时（python ingest/embedder.py）项目根不在 sys.path，
# 把它加进去，否则 import 不到 config 包。
# 作为包被 import 时（from ingest.embedder import ...）__package__ 非空，跳过。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from config.settings import settings

# 通义千问 OpenAI 兼容接口地址（Embedding 也走这套协议）
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模块级单例：整个项目共用一个 client，避免重复初始化
_client = OpenAI(
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量把文本转成向量，返回 [[dim 个 float], ...]。

    注意：批量大小由调用方控制（build_index 里每批 64 条），
    防止单次请求过长触发 API 限流/超时。
    resp.data 与 input 顺序一致，按 index 排序是保险写法。
    """
    if not texts:
        return []
    resp = _client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def embed_query(text: str) -> list[float]:
    """把单个查询问题转成向量（检索用，query 不走批量接口）。"""
    resp = _client.embeddings.create(model=settings.EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def probe_dimension(sample: str = "向量维度探测") -> int:
    """探明模型实际输出的向量维度（本项目模型实测 1024 维）。

    为什么必须探？Collection 的 dim 建好后固定，写入不同维度的向量会报错。
    先跑一次真实调用拿到 len(vec)，再拿这个值去建表，是工程上的稳妥顺序。
    """
    vec = embed_texts([sample])[0]
    return len(vec)


if __name__ == "__main__":
    dim = probe_dimension()
    print(f"模型 {settings.EMBEDDING_MODEL} 输出维度: {dim}")
