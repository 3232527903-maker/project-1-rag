"""
查看向量库数据小工具（Day 4 辅助，PyCharm 里右键运行即可）

用法（项目根目录下）：
    .venv/bin/python view_data.py                   # 看前 5 条
    .venv/bin/python view_data.py --limit 10        # 看前 10 条
    .venv/bin/python view_data.py --title "账单"     # 只看某一篇文档
    .venv/bin/python view_data.py --all-fields      # 连向量一起显示（前几条）
"""

import argparse
import sys
from pathlib import Path

# 直接运行时把项目根加进 sys.path（保证 import 到 config / ingest）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.vector_store import COLLECTION_NAME, _make_client


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 Milvus 向量库里的数据")
    parser.add_argument("--limit", type=int, default=5, help="显示条数（默认 5）")
    parser.add_argument("--title", type=str, default="", help="按文档标题筛选，如 --title 账单")
    parser.add_argument("--all-fields", action="store_true", help="连向量一起显示（默认只显示文本字段）")
    args = parser.parse_args()

    client = _make_client()
    if not client.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 不存在，请先跑 .venv/bin/python ingest/build_index.py 入库")
        return

    # Milvus 查询前必须先 load_collection（把集合加载进内存），否则报 code=101
    client.load_collection(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"集合 {COLLECTION_NAME} 总行数: {stats.get('row_count')}")

    # 组装查询：字段 + 可选筛选条件 + 条数
    fields = ["title", "chunk_index", "source", "text"]
    if args.all_fields:
        fields.append("vector")
    filter_expr = f'title == "{args.title}"' if args.title else ""

    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter=filter_expr,
        output_fields=fields,
        limit=args.limit,
    )
    print(f"显示 {len(rows)} 条：")
    for i, r in enumerate(rows, 1):
        print(f"\n--- 第 {i} 条 ---")
        print(f"  标题    : {r.get('title')}")
        print(f"  段落序号: {r.get('chunk_index')}")
        print(f"  来源    : {r.get('source')}")
        if args.all_fields:
            vec = r.get("vector", [])
            print(f"  向量    : [{', '.join(f'{x:.3f}' for x in vec[:8])}...] 共 {len(vec)} 维")
        print(f"  正文    : {r.get('text', '')[:80]}")


if __name__ == "__main__":
    main()
