# 技术笔记 · 踩坑记录（面试素材库）

> 原则：每条按「现象 → 根因 → 解决 → 面试怎么说」整理。会讲坑 = 真做过的人。
> 更新：Day 6（2026-09-01）

---

## 1. `.env` 键名不能叫 `MILVUS_URI`（pymilvus 冲突）

- **现象**：跑代码报 `Illegal uri: ./xxx.db`，加载 .env 时崩溃。
- **根因**：pymilvus 3.x 在 `import` 时会自动读取 `.env` 里的环境变量，把 `MILVUS_URI` 当成连接地址解析；我们的值 `./milvus_lite.db` 是本地文件路径，被当远程地址解析就崩了。
- **解决**：改用自定义键名 `MILVUS_LOCAL_DB`，只供本项目代码读取，避开 pymilvus 的环境变量扫描。
- **面试怎么说**："我踩过 pymilvus 自动加载 .env 的坑——所以环境变量键名要和第三方库的保留键错开。"

## 2. langchain-openai 的 OpenAIEmbeddings 与 DashScope 兼容接口不兼容

- **现象**：调用 embedding 报 400，请求格式不对。
- **根因**：langchain-openai 1.6 发出的请求体格式（OpenAI 标准）和 DashScope 的 OpenAI 兼容接口实际要求的格式不一致。
- **解决**：弃用 langchain 封装，改用 openai SDK 直调（`openai.OpenAI(base_url=dashscope)`）。
- **面试怎么说**："兼容接口 ≠ 完全兼容，框架封装层会做自己的序列化，遇到 400 时直接看请求体，必要时绕开封装直调 SDK。"

## 3. DashScope embedding 单批上限 20

- **现象**：全量入库时 batch size=64 报 400。
- **根因**：DashScope text-embedding-v3 单次请求最多 20 条文本。
- **解决**：`BATCH_SIZE` 从 64 改为 20，循环分批。
- **面试怎么说**："写批量脚本前先看 API 的批次限制，不要想当然；报错时先怀疑参数边界。"

## 4. Milvus 查询前必须先 `load_collection()`

- **现象**：查询报 `code=101`。
- **根因**：Milvus 集合要先加载进内存才能查询。
- **解决**：`client.load_collection(COLLECTION_NAME)` 后再 query/search。
- **面试怎么说**："Milvus 的查询前置条件是集合已加载，这个状态错（code=101）是高频坑。"

## 5. `user_prompt` 误写成列表 → DeepSeek 400（Day 5 自己踩的）

- **现象**：`BadRequestError 400: messages[1]: invalid type: string ... expected ChatCompletionRequestContentBlock`。
- **根因**：`user_prompt = [f"..."]` 误用**方括号**成了列表 → openai SDK 把 content 序列化成裸字符串数组 `["..."]`，DeepSeek 服务端要求数组元素必须是 `{"type":"text","text":...}` 对象 → 400。
- **解决**：`user_prompt` 改回**字符串**。
- **面试怎么说**："同一段 code 用方括号包了一下就 400——content 要么是字符串、要么是规范的内容块数组，不能是裸字符串数组。这类类型坑直接看报错里的 expected 字段。"

## 6. 向量库路径用相对路径 → 换工作目录就找不到库

- **现象**：PyCharm 右键运行 `search.py` 报「Collection 不存在」，但终端跑就正常。
- **根因**：`MILVUS_LOCAL_DB` 默认值 `./milvus_lite.db` 依赖「当前工作目录」；PyCharm 以 `retrieval/` 为工作目录运行时去 `retrieval/milvus_lite.db` 找库，自然找不到。
- **解决**：默认值改为**基于 `BASE_DIR` 的绝对路径** `str(BASE_DIR / "milvus_lite.db")`。
- **面试怎么说**："文件路径一律基于项目根（BASE_DIR）拼绝对路径，不依赖运行入口的工作目录——否则换 IDE / 换启动方式就玄学报错。"

---

## 工程化要点速记（Day 6）

- **异常处理分级**：可重试（超时 / 连接 / 429 限流 → 指数退避）vs 不可重试（401 / 400 → 直接抛）。重试必须有上限，防止雪崩。
- **单测三原则**：快（秒级）、稳（不依赖网络/时间/环境）、隔离（mock 外部依赖，只测自己逻辑）。
- **monkeypatch**：pytest 把函数替换成假实现，验证"调对了接口、组装对了数据"。
