"""
分块层（Day 3 实现）

职责：把 loader 产出的 Document 切成「长度适中、语义完整」的 chunk，
供 Day 4 向量化写入 Milvus。

为什么分块是面试考点：
- chunk 是「检索的最小单位」，检索命中后返回的就是它；
- 切太大 → 语义被稀释、定位不精确；切太小 → 语义被切断、上下文丢失；
- overlap 保留相邻块的重叠，避免关键信息被拦腰切断。
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 分块参数（第 1 周先固定，第 2 周再做对比实验）
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

@dataclass
class Chunk:
    """一个分块。text 是正文，source/title 继承自原文档，chunk_index 是块序号。"""
    text: str
    source: str = ""
    title: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict:
        """转成字典，方便后续写入 Milvus。"""
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
        }


def _make_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """造一个 RecursiveCharacterTextSplitter。

    separators 按「大边界 → 小边界」排序：优先在段落/换行/句子标点处切，
    实在不行才硬切单个字符，尽量避免切断语义。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )

def chunk_document(doc, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """把一篇 Document 切成多个 Chunk"""
    splitter = _make_splitter(chunk_size, chunk_overlap)
    texts = splitter.split_text(doc.text)   # -> list[str]
    return [
        Chunk(text=t, source=doc.source, title=doc.title, chunk_index=i)
        for i, t in enumerate(texts)
    ]


def chunk_all(docs, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """批量分块：遍历所有文档，汇总成一个 Chunk 列表。"""
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 直接跑「python ingest/chunker.py」时，sys.path[0] 是脚本目录 ingest/，
    # 找不到上层的 ingest 包，所以手动把项目根目录加进 sys.path（和 tests 里同款写法）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from ingest.loader import load_all

    docs = load_all()
    chunks = chunk_all(docs)
    lens = [len(c.text) for c in chunks]
    print(f"共 {len(docs)} 篇 → {len(chunks)} 个 chunk")
    print(f"平均长度 {sum(lens) / len(lens):.0f} 字，最长 {max(lens)} 字")
    print("前 3 个 chunk 示例:")
    for c in chunks[:3]:
        print(f"- [{c.title}] #{c.chunk_index} ({len(c.text)} 字): {c.text[:30]}...")