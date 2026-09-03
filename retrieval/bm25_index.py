"""
BM25 稀疏索引（Day 8 实现）

职责：输入全部 chunk 文本 -> jieba 分词 -> 构建 BM25Okapi -> 查询时返回 Top-K 关键词命中片段。

实现要点（混合检索第一步，给 RRF 融合提供「关键词路」）：
- 中文必须先分词：BM25 按词项（token）统计，不分词的话 query 与文档字面无法对齐
- 输入与入库同源（text/source/title/chunk_index），保证能和向量库按 source|chunk_index 对齐
- 返回片段带 bm25_score，RRF 只看名次、不直接用分数
"""

# TODO(Day 8)：实现 BM25 稀疏索引，例如：
# from rank_bm25 import BM25Okapi
# import jieba
#
# class BM25Index:
#     def __init__(self, docs: list[dict]):
#         """docs: [{text, source, title, chunk_index}, ...]（与入库同源）"""
#         ...
#     def search(self, query: str, top_k: int = 5) -> list[dict]:
#         """分词 query -> get_scores -> 降序取 top_k，返回含 bm25_score 的片段列表"""
#         ...
import jieba
from pathlib import Path
from rank_bm25 import BM25Okapi
def _tokenize (text:str) ->list[str]:
    return list(jieba.cut(text))

class BM25Index:
    def __init__(self,docs:list[dict]):
        self.docs = docs
        tokenized =[_tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
    def search(self,query:str,top_k:int=5) ->list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
        return [{
            **self.docs[i],"bm25_score":round(float(scores[i]),4)
        }for i in ranked[:top_k]
        ]
if __name__ == '__main__':
    from ingest.loader import load_all

    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    docs = load_all(RAW_DIR)
    docs_dicts = [d.to_dict() for d in docs]
    # print(docs[:1])
    bm25 = BM25Index(docs=docs_dicts)
    res = bm25.search(query='怎么调用api')
    print(len(res))
    for i in res:
        print(i)