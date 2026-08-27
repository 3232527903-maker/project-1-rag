"""
抓取 WorkBuddy 帮助中心文档，转成干净 Markdown 存入 data/raw/。

数据源：https://www.workbuddy.cn/docs/sitemap-workbuddy-workbuddy.xml
站点：VitePress，正文位于 <main class="main"> -> <div class="vp-doc">。

用法（在项目根目录下执行）：
    python ingest/fetch_docs.py

输出：
    data/raw/ 下按目录结构存放 .md 文件，每篇头部含来源 URL + 标题元数据。
"""

import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ===== 抓取配置 =====
SITEMAP_URL = "https://www.workbuddy.cn/docs/sitemap-workbuddy-workbuddy.xml"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
DELAY = 0.3      # 每篇间隔秒数，礼貌抓取
TIMEOUT = 15     # 单篇超时（秒）
MAX_RETRY = 2    # 失败重试次数

# data/raw 目录（本文件位于 ingest/，上两级是项目根）
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_sitemap_urls() -> list[str]:
    """抓取 sitemap，返回所有 <loc> 里的 URL。"""
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]


def url_to_relpath(url: str) -> str:
    """把页面 URL 转成 data/raw 下的相对路径（保留目录结构，统一 .md 后缀）。"""
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path).strip("/")

    # 去掉 /docs/workbuddy/ 前缀
    if path.startswith("docs/workbuddy/"):
        path = path[len("docs/workbuddy/"):]
    elif path == "docs/workbuddy":
        path = ""

    # 首页特殊处理为 index.md
    if not path:
        return "index.md"
    if not path.endswith(".md"):
        path += ".md"
    # 文件名里的空格换成连字符，避免路径带空格
    return path.replace(" ", "-")


def extract_content(html: str) -> tuple[str, str]:
    """从 VitePress 页面 HTML 提取 (标题, 正文 markdown)。"""
    soup = BeautifulSoup(html, "lxml")

    # 正文容器：main -> .vp-doc；找不到再逐级 fallback
    main = soup.find("main")
    doc = None
    if main is not None:
        doc = main.find("div", class_=re.compile(r"vp-doc"))
    if doc is None:
        doc = soup.find("div", class_=re.compile(r"vp-doc"))
    if doc is None:
        doc = main
    if doc is None:
        return "", ""

    # 标题（正文第一个 <h1>）
    h1 = doc.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # 去掉 VitePress 的标题锚点链接（# 号小图标），避免混进正文
    for anchor in doc.find_all("a", class_="header-anchor"):
        anchor.decompose()

    body_md = md(str(doc), heading_style="ATX")
    return title, body_md


def fetch_one(url: str) -> tuple[str, str] | None:
    """抓取单篇，返回 (title, markdown)；失败返回 None。"""
    last_err = None
    for _ in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return extract_content(resp.text)
        except Exception as e:  # noqa: BLE001 抓取脚本需要兜住所有网络异常
            last_err = e
            time.sleep(1)
    print(f"  [失败] {url}: {last_err}")
    return None


def main() -> None:
    print("抓取 sitemap:", SITEMAP_URL)
    urls = fetch_sitemap_urls()
    total = len(urls)
    print(f"共 {total} 个页面\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for i, url in enumerate(urls, 1):
        rel = url_to_relpath(url)
        target = RAW_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        result = fetch_one(url)
        if result is None:
            fail += 1
            continue
        title, body_md = result

        header = f"<!--\nsource: {url}\ntitle: {title}\n-->\n\n"
        target.write_text(header + body_md, encoding="utf-8")
        ok += 1
        print(f"[{i}/{total}] {rel}  <- {title or '(无标题)'}")
        time.sleep(DELAY)

    print(f"\n完成：成功 {ok} 篇，失败 {fail} 篇，输出目录 {RAW_DIR}")


if __name__ == "__main__":
    main()
