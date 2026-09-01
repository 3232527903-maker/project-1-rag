"""
Day 6 单测：验证 llm 的 _build_context 纯函数（不依赖网络）。

运行方式（项目根目录下）：
    .venv/bin/python -m pytest tests/test_llm.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generation.llm import _build_context


def test_build_context_编号来源与顺序():
    """多个上下文应按顺序编号、拼接来源。"""
    contexts = [
        {"title": "文档A", "source": "http://a", "text": "内容1"},
        {"title": "文档B", "source": "http://b", "text": "内容2"},
    ]
    out = _build_context(contexts)
    assert "[1] 来源：文档A（http://a）" in out
    assert "[2] 来源：文档B（http://b）" in out
    assert out.index("内容1") < out.index("内容2")  # 顺序保持


def test_build_context_empty():
    """空上下文应返回空字符串。"""
    assert _build_context([]) == ""


def test_build_context_single():
    """单个上下文：编号从 1 开始。"""
    out = _build_context([{"title": "T", "source": "http://s", "text": "正文"}])
    assert "[1] 来源：T（http://s）" in out
    assert "正文" in out
