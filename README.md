# 项目一 · WorkBuddy 帮助中心 RAG 问答系统

基于 WorkBuddy 帮助中心文档构建的智能知识库问答系统（RAG）。

> 状态：Day 1 骨架已搭建，功能待后续逐日实现。

## 技术栈

- 数据源：WorkBuddy 帮助中心文档（VitePress 站）
- 框架：LangChain
- 向量库：Milvus（Milvus Lite）
- 生成：DeepSeek（OpenAI 兼容接口）
- Embedding：通义 text-embedding-v3
- 服务：FastAPI（第 4 周封装）

## 目录结构

```
project-1-rag/
├── config/        # 配置（settings.py 读取 .env）
├── ingest/        # 数据接入层（文档解析，Day 2）
├── retrieval/     # 检索层（向量检索，Day 5）
├── generation/    # 生成层（LLM 调用，Day 5）
├── data/raw/      # 原始文档（WorkBuddy 文档）
└── tests/         # 单元测试
```

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DeepSeek / 通义的 API Key

# 4. 验证配置读取
python -c "from config.settings import settings; print(settings.DEEPSEEK_MODEL)"
```

## 进度

- [x] Day 1：项目初始化 + 环境搭建
- [ ] Day 2：数据接入与文档解析
- [ ] Day 3：分块 Chunking
- [ ] Day 4：向量化 + 写入 Milvus
- [ ] Day 5：检索 + 生成（最小闭环）
- [ ] Day 6：联调 + 测试 + 记录踩坑
- [ ] Day 7：周复盘 + 优化清单
