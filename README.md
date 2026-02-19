使用 LlamaIndex（支持多索引 + 路由）实现多场景或多领域 RAG 知识隔离架构。

## 0. 阶段汇总（本项目已完成）

### Phase 1：配置持久化 + API 骨架
- 新增 `app_config.json` 配置持久化
- 提供 `settings/scenes/chat/rebuild` API
- 支持服务启动后自动加载场景配置

### Phase 2：Web 可视化页面
- 提供聊天页、设置页、场景管理页
- 支持场景增删改、保存与重建索引

### Phase 3：稳定性与重建任务优化
- 索引重建改为后台任务（`task_id` + 状态轮询）
- 重建期间聊天返回明确提示，避免长时间阻塞
- 提供重建任务状态卡片（进度、状态、消息）
- 支持单场景增量重建和全量重建
- 全量重建默认仅处理“发生变化”的场景（基于场景配置 + embedding 配置 + 文件签名）
- 新增场景文档目录 `storage/scene_catalog.json`（用于确定性统计问答）
- 统计/列表类问题（如“有几份文档”“列出文档”）走确定性回答，不再依赖 RAG 片段覆盖

### Phase 4：桌面化与安装包
- 新增 `desktop_main.py` 桌面入口（内置本地服务）
- 新增 `build_desktop.py` 打包脚本（PyInstaller）
- 新增 Inno Setup 脚本 `installer/multi_scene_rag_setup.iss`

### 当前模型配置能力（重点）
- LLM：使用 OpenAI 兼容接口，可切换多种厂商（`llm_vendor`）
  - `dashscope`（通义千问）
  - `openai`
  - `deepseek`
  - `claude` / `gemini` / `glm` / `kimi` / `custom`（需自填 base_url）
- Embedding：支持三类 provider
  - `dashscope`
  - `openai`
  - `local`（本地模型）
- 本地 embedding 支持三种来源（`embedding_source`）
  - `huggingface`
  - `modelscope`
  - `local`（本地目录）
- 设备选择（`embedding_device`）：`cpu` / `cuda`（cuda 不可用自动回退 cpu）
- 人性化提醒：修改 embedding 相关配置后会提示“建议全量重建索引”

## 1. 项目结构与核心模块

```
multi_scene_rag/
├── app.py              # CLI 入口（交互式问答）
├── api_server.py       # Web API 服务（FastAPI）
├── desktop_main.py     # 桌面应用入口（PyWebView + 内嵌服务）
├── build_desktop.py    # PyInstaller 打包脚本
├── config.py           # 默认场景定义（hr / it / finance）
├── config_store.py     # 配置持久化（app_config.json）
├── classifier.py       # 场景路由（关键词 + LLM 兜底）
├── rag_engine.py       # 核心 RAG 引擎（MultiSceneRAG）
├── query_orchestrator.py   # 查询意图路由（独立模块，用于 count/list/rag 分流）
├── query_strategy.py   # 查询策略与确定性回答规则
├── scene_catalog.py    # 场景文档目录管理（扫描、签名、确定性统计）
├── web/
│   ├── index.html      # 前端页面
│   └── static/         # 静态资源（app.js, styles.css）
├── data/               # 场景文档目录（hr, it, finance）
├── storage/            # 向量索引、场景目录、重建清单
├── scripts/            # 回归测试脚本
├── tests/              # 单元测试
└── installer/          # Inno Setup 安装脚本
```

**核心数据流**：用户问题 → `classify_scene()` 路由 → 意图识别（count/list/qa）→ 统计类走 `scene_catalog` 确定性回答，问答类走 RAG → LLM 生成回答

## 2. 环境准备

```bash
conda create -n multi-scene-rag-workflow python=3.12.9 -y
conda activate multi-scene-rag-workflow
pip install -r requirements.txt
```

## 3. 运行方式

### 3.1 命令行模式

```bash
python app.py
```

### 3.2 Web 模式（推荐）

```bash
python api_server.py
```

访问：

```text
http://127.0.0.1:8000/
```

页面功能：
- 聊天问答（含 answer_mode、usage 展示）
- API Key / Base URL / LLM 厂商（llm_vendor）/ 模型名称 设置
- Embedding Provider / Embedding 模型设置
- Embedding Source（huggingface / modelscope / local）
- Embedding Device（cpu / cuda，cuda 不可用会自动回退 cpu）
- 本地 Embedding 模型状态检查与下载（provider=local 时）
- 场景增删改
- 单场景增量重建 / 全量重建
- 重建任务状态卡片（任务 ID、状态、进度）
- 统计/列表类问法的确定性回答（跨场景一致）
- 聊天回答返回 `answer_mode`（structured / rag）、`usage`（token 消耗）

## 4. API 接口一览

| 方法 | 路径 | 说明 |
|-----|------|------|
| GET | `/health` | 健康检查 |
| GET | `/` | 前端页面 |
| GET | `/api/settings` | 获取当前配置（API Key 掩码） |
| PUT | `/api/settings` | 保存配置（API Key、LLM 厂商、Base URL、模型、Embedding 等） |
| GET | `/api/embedding-models` | 获取 Embedding 模型选项 |
| GET | `/api/embedding-capabilities` | 获取 Embedding 能力配置 |
| POST | `/api/embedding/model-status` | 检查本地 embedding 模型状态 |
| POST | `/api/embedding/model-download` | 启动本地 embedding 模型下载（后台任务） |
| GET | `/api/embedding/model-download-status/{task_id}` | 查询模型下载任务状态 |
| GET | `/api/scenes` | 获取场景列表 |
| PUT | `/api/scenes` | 保存场景配置 |
| POST | `/api/chat` | 聊天问答 |
| POST | `/api/rebuild-index` | 启动索引重建（单场景或全量） |
| GET | `/api/rebuild-status/{task_id}` | 查询重建任务状态 |

## 5. 本地配置说明

首次启动 `app.py` / `api_server.py` / `desktop_main.py` 时，会在项目根目录自动生成 `app_config.json`。

`app_config.json` 持久化：
- `api_key`
- `base_url`
- `llm_vendor`（LLM 厂商）
- `model_name`
- `embedding_provider`
- `embedding_model`
- `embedding_source`
- `embedding_device`
- `scenes`（动态场景配置）

配置优先级（双模式）：
- 桌面/Web 模式（`desktop_main.py` / `api_server.py`）：`app_config.json` > 环境变量 > 默认值
- 命令行模式（`app.py`）：环境变量 > `app_config.json` > 默认值

Embedding 模型缓存目录：
- HuggingFace：`embedding/huggingface/`
- ModelScope：`embedding/modelscope/`

缓存行为：
- 首次使用会下载模型到上述目录
- 后续同模型优先读取本地缓存，不重复下载
- 仅在更换模型、清理缓存或版本变化时才会再次下载

## 6. 桌面应用（Phase 4）

### 6.1 直接启动桌面版（开发调试）

```bash
python desktop_main.py
```

### 6.2 构建桌面程序（PyInstaller）

> 说明：`pyinstaller` 是**打包工具依赖**，默认不放在 `requirements.txt`（运行依赖）中。

先安装打包工具：

```bash
python -m pip install pyinstaller
python -m PyInstaller --version
```

执行构建：

```bash
python build_desktop.py
```

输出目录：

```text
dist/MultiSceneRAG/
```

主程序：

```text
dist/MultiSceneRAG/MultiSceneRAG.exe
```

### 6.3 生成 Win10/Win11 安装包（Inno Setup）

使用 Inno Setup 打开并编译脚本：

```text
installer/multi_scene_rag_setup.iss
```

安装包输出目录：

```text
dist_installer/
```

## 7. 辅助能力

### 7.1 本地 Embedding 模型下载

当 `embedding_provider=local` 时，可通过 Web 设置页或 API 触发模型下载：
- **HuggingFace**：自动下载到 `embedding/huggingface/<model_id>/`
- **ModelScope**：自动下载到 `embedding/modelscope/`，路径记录在 `embedding/download_registry.json`
- 下载为后台任务，可通过 `GET /api/embedding/model-download-status/{task_id}` 轮询状态

### 7.2 回归测试脚本

| 脚本 | 说明 |
|-----|------|
| `scripts/regression_full_suite.py` | 全量回归：count/list/qa 各场景、跨场景路由、answer_mode 校验、token 消耗 |
| `scripts/regression_structured_queries.py` | 结构化查询专项（确定性答案、RAG 分流） |
| `scripts/regression_api_chat.py` | API 聊天接口回归 |

### 7.3 单元测试

```bash
python -m unittest discover -s tests -p "test_*.py"
# 或
python -m pytest tests/ -v
```

覆盖模块：`query_orchestrator`、`query_strategy`、`scene_catalog`。

### 7.4 端口冲突排查

若 8000 端口被占用，参见 `port_8000_fix.md`。桌面版可通过环境变量 `MSRAG_PORT` 指定端口。

## 8. 常见问题

### 8.1 `No module named PyInstaller`

原因：当前终端不是安装 `pyinstaller` 的 Python/Conda 环境。  
处理：先 `conda activate multi-scene-rag-workflow`，再执行安装命令。

### 8.2 打包后页面空白或打不开

优先检查（参见 `port_8000_fix.md` 排查端口占用）：
- `web/` 是否被正确打包
- `data/` 是否被正确打包
- 端口（默认 `8000`）是否被占用

### 8.3 更换 embedding 模型后检索效果异常

原因：向量库中的历史向量和当前 embedding 配置不一致。  
处理：执行一次“重建索引（全量）”，保证入库向量和查询向量使用同一 embedding 配置。

### 8.4 本地 embedding 首次加载很慢

原因：首次会下载模型到本地缓存目录。  
处理：属正常现象；后续同模型会复用缓存，不会重复下载。

### 8.5 为什么“知识库有几份文档”有时回答不稳定

原因：这类问题属于统计/清单问法，不应依赖语义检索片段。  
处理：当前版本已将“数量/列表”类问题切换为确定性回答，数据来源为 `storage/scene_catalog.json`。如结果异常，先执行一次重建（会自动刷新目录）。

## 9. 关键操作清单（给交付/运维）

1) 安装依赖
```bash
conda activate multi-scene-rag-workflow
pip install -r requirements.txt
```

2) 启动 Web 服务
```bash
# 脚本语法检查
python -m py_compile api_server.py
python api_server.py
```

3) 首次配置建议
- 先配置 `api_key/base_url/model_name`
- 再配置 `embedding_provider/embedding_model/embedding_source/embedding_device`
- 修改 embedding 后立即执行“重建索引（全量）”

4) 打包桌面版
```bash
python -m pip install pyinstaller
python build_desktop.py
```

5) 生成安装包
- 使用 Inno Setup 编译：`installer/multi_scene_rag_setup.iss`

6) 回归测试（推荐）
```bash
python -m unittest discover -s tests -p "test_*.py"
```

