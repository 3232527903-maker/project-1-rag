"""
配置模块：统一从 .env 读取所有配置项。

为什么要单独一个 config 模块？
1. 密钥和配置集中管理，代码里不硬编码（API Key 一旦写死在代码里，提交就泄露）。
2. 换环境（本地 / 服务器）只需改 .env，不改代码。
3. 面试常问「配置怎么管理」，这就是标准答案：环境变量 + .env + dotenv 加载。

使用方式：
    from config.settings import settings
    print(settings.DEEPSEEK_MODEL)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = 本文件的上两级目录（config/settings.py -> 项目根）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载项目根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")


class Settings:
    """集中管理配置项。

    每个属性对应 .env 里的一个变量。用 os.getenv(key, 默认值) 读取，
    好处是：即使 .env 里某个变量漏填，也有默认值兜底，程序不会直接崩。
    """

    # ===== 生成模型（DeepSeek）=====
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    # ===== Embedding（通义 qwen3.7-text-embedding）=====
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")

    # ===== 向量库（Milvus Lite）=====
    # 注意：不能用键名 MILVUS_URI——pymilvus 3.x import 时会自动加载 .env，
    # 读到本地文件路径会把它当远程地址解析而崩溃（Illegal uri: ./xxx.db）。
    # 自定义 MILVUS_LOCAL_DB 只供本项目读取，避免与 pymilvus 的环境变量冲突。
    MILVUS_LOCAL_DB: str = os.getenv("MILVUS_LOCAL_DB", "./milvus_lite.db")


# 模块级单例：整个项目 import 同一个 settings 实例，避免重复加载 .env
settings = Settings()
