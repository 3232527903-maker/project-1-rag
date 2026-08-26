"""
数据接入层（Day 2 实现）

职责：把 WorkBuddy 帮助中心的原始文档（Markdown / HTML）解析成干净的纯文本，
并保留来源、标题等元数据，供后续分块使用。

计划实现：
- 支持 Markdown / HTML（WorkBuddy 是 VitePress 站，正文是 Markdown 转 HTML）
- 清洗：去导航、去页眉页脚、去多余空行、去乱码字符
- 保留元数据：source（来源 URL）、title（页面标题）
"""

# TODO(Day 2)：实现文档解析函数，例如：
# def parse_document(raw_path: str) -> tuple[str, dict]:
#     """解析单个文档，返回 (纯文本, 元数据字典)。"""
#     raise NotImplementedError("Day 2 实现")
