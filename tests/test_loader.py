"""
Day 2 单测：验证 loader 解析结果「干净可用」。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_loader.py -v
"""

import sys
from pathlib import Path

# 让测试能 import 项目根下的 ingest 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.loader import clean_text, load_all, parse_markdown


def test_clean_text_removes_images():
    """图片语法应被去掉。"""
    text = "看图 ![alt text](/docs/static/x.png) 结束"
    cleaned = clean_text(text)
    assert "![alt text]" not in cleaned
    assert "结束" in cleaned


def test_clean_text_removes_invisible_chars():
    """零宽字符应被去掉。"""
    assert "\u200b" not in clean_text("账\u200b单")


def test_clean_text_collapses_blank_lines():
    """连续 3 个空行应压成 1 个空行。"""
    assert "\n\n\n\n" not in clean_text("a\n\n\n\nb")
    assert "a\n\nb" == clean_text("a\n\n\n\nb")


def test_parse_markdown_extracts_metadata():
    """头部元数据应被正确提取，标题里的零宽字符应被清理。"""
    content = "<!--\nsource: https://example.com/doc\ntitle: 账\u200b单\n-->\n\n# 账单\n\n正文内容"
    doc = parse_markdown(content)
    assert doc.source == "https://example.com/doc"
    assert doc.title == "账单"          # 零宽字符被去掉
    assert "正文内容" in doc.text


def test_load_all_returns_documents():
    """批量加载应读到 84 篇有内容的文档（85 篇里首页为空被跳过），每篇都有 source。"""
    docs = load_all()
    assert len(docs) >= 84
    for d in docs:
        assert d.source.startswith("https://")
        assert d.text  # 正文不为空
