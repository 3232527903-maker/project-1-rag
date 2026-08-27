"""
数据接入层（Day 2 实现）

职责：把 data/raw/ 下的原始文档解析成「干净的纯文本 + 元数据」，供 Day 3 分块使用。

当前实现 Markdown 解析（数据源是 WorkBuddy 帮助中心抓取的 .md 文件）；
PDF / Word 解析预留了分发接口（当前数据源没有这两种格式）。

解析做了三件事：
1. 提取元数据：从每篇文件头部的 <!-- source / title --> 注释里读出来源 URL 与标题。
2. 清洗正文：去掉图片语法、零宽字符、多余空行等「脏东西」。
3. 结构化输出：统一成 Document 对象（text + source + title）。

为什么「解析要干净」是面试考点：
- 数据预处理质量直接决定后续检索质量（垃圾进、垃圾出）。
- 面试常问「文档解析做了什么清洗」，答案就是下面 clean_text 里的几件事。
"""

import re
from dataclasses import dataclass
from pathlib import Path

# 项目根 = 本文件上两级（ingest/loader.py -> 项目根）
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# 零宽字符等不可见字符（VitePress 抓取时混进来的，肉眼看不见但会污染文本）
_INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff\u00a0]")

# 图片语法：![alt](url) 或 ![](url) —— RAG 只处理文本，图片是站内相对路径、本地也失效，直接去掉
_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# 头部元数据注释：<!-- source: ... \n title: ... -->
_META_PATTERN = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)


@dataclass
class Document:
    """一篇解析后的文档。"""
    text: str          # 清洗后的纯文本
    source: str = ""   # 来源 URL
    title: str = ""    # 标题
    path: str = ""     # 原始文件路径

    def to_dict(self) -> dict:
        """转成字典，方便后续（写入 Milvus 时用）。"""
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "path": self.path,
        }


def clean_text(text: str) -> str:
    """清洗文本：去图片、去零宽字符、压缩多余空行。"""
    text = _IMAGE_PATTERN.sub("", text)      # 去掉图片语法
    text = _INVISIBLE_CHARS.sub("", text)    # 去掉零宽/不可见字符
    text = re.sub(r"[ \t]+\n", "\n", text)   # 去掉行尾空白
    text = re.sub(r"\n{3,}", "\n\n", text)   # 连续空行压成 1 个空行
    return text.strip()


def _extract_metadata(content: str) -> tuple[dict, str]:
    """拆出头部 <!-- ... --> 元数据，返回 (元数据字典, 去掉头部的正文)。"""
    meta: dict = {}
    body = content
    m = _META_PATTERN.match(content.lstrip())  # 只在文件开头匹配头部注释
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        body = content[m.end():]
    return meta, body


def parse_markdown(content: str, source_path: str = "") -> Document:
    """解析单篇 Markdown 内容，返回 Document。"""
    meta, body = _extract_metadata(content)
    text = clean_text(body)
    title = _INVISIBLE_CHARS.sub("", meta.get("title", "")).strip()
    return Document(
        text=text,
        source=meta.get("source", ""),
        title=title,
        path=source_path,
    )


def load_document(path: Path) -> Document:
    """按扩展名加载单个文档文件。当前实现 .md，PDF / Word 预留。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path.read_text(encoding="utf-8"), str(path))
    # TODO: 后续如需支持 PDF / Word，在这里按扩展名分发：
    #   suffix == ".pdf"  -> parse_pdf(path)   (用 PyMuPDF)
    #   suffix == ".docx" -> parse_docx(path)  (用 python-docx)
    raise ValueError(f"暂不支持的文件类型: {suffix}")


def load_all(raw_dir: Path = RAW_DIR, skip_empty: bool = True) -> list[Document]:
    """遍历 raw_dir 下所有 .md 文件，批量解析，返回 Document 列表。

    skip_empty=True 时跳过正文为空的文档（如首页纯导航页，无检索价值）。
    """
    docs = []
    for path in sorted(raw_dir.rglob("*.md")):
        doc = load_document(path)
        if skip_empty and not doc.text:
            continue
        docs.append(doc)
    return docs


if __name__ == "__main__":
    docs = load_all()
    print(f"共解析 {len(docs)} 篇文档\n")
    total_chars = sum(len(d.text) for d in docs)
    print(f"总字符数: {total_chars}")
    print("前 3 篇示例:")
    for d in docs[:3]:
        print(f"- [{d.title}] {d.source} ({len(d.text)} 字)")
