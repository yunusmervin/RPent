# RPent 多模态记忆系统重构 · 总体设计

版本：v1.0 ｜ 对应任务卡：`tasks/T0.md` – `tasks/T8.md`

## 0. 背景与目标

RPent 现有记忆系统是一套带 YAML frontmatter 的 Markdown 文件语料库
（`rpent/memory/manager.py`），承担语义记忆（global/task-family 叶子）、
程序性记忆（task-specific 的 audit.json + recipe.jsonl）、证据学
（confidence 晋级、冲突隔离）、HF 分发四项职责。

本次重构在不破坏以上机制的前提下新增两块能力：

1. **多模态情景记忆与向量召回**：episode 级记录（任务语言 + 关键帧 +
   结果 + 关联知识叶子）向量化，支持"用当前画面反查历史相似场景"；
2. **会话历史与偏好记忆**：TUI/Dashboard 交互、人工 verdict、操作者偏好
   的结构化管理与召回。

非目标：不改变 merge 事务语义；不改变 Flash Mode 回放格式；不替换
Markdown 语料的事实源地位。

## 1. 总体架构：三平面

```
┌─ 平面 K：知识语料层（现有，唯一事实源）────────────────────────┐
│  memory/<robot>/{global, task-family, task-specific, results} │
│  + merge 事务 + 可信度晋级 + 冲突隔离 + HF 分发                  │
└──────────────▲────────────────────────────────┬──────────────┘
               │ 派生（可重建）                  │ merge 完成钩子
┌─ 平面 E：多模态情景记忆（新建）─────────────────┴──────────────┐
│  AGFS /frames/：关键帧、overlay（SHA256 内容寻址）              │
│  AGFS /episodes/：episode.json（全量元数据，随语料分发）        │
│  Milvus：向量索引（collection-per-version，partition_key=robot）│
│  embedding_server：Qwen3-VL-Embedding-2B 独立进程              │
└────────────────────────────▲─────────────────────────────────┘
┌─ 平面 S：会话与偏好（新建，mem0 开源实例，向量后端=Milvus）─────┐
│  run_id=cell_tag，user_id=操作者                               │
│  内容：任务指令、TUI 引导、人工 verdict、操作者偏好              │
└───────────────────────────────────────────────────────────────┘
         三个平面统一通过 MCP 工具暴露给 Agentic Planner
```

## 2. 接口契约

### 2.1 MemoryBackend（T0）

```python
class MemoryBackend(Protocol):
    """记忆后端协议。FileSystemBackend 为默认实现。"""

    @property
    def root(self) -> Path: ...

    def authorize_read(self, path: str | Path) -> Path: ...
    def authorize_write(self, path: str | Path) -> Path: ...
    def list_directory(self, path: str | Path) -> tuple[Path, list[str] | None]: ...
    def record_read(self, path: Path, content: str) -> None: ...

    def merge_memory(self, *, cell_tag: str, run_state_dir: str | Path,
                     solved: bool) -> dict[str, Any]: ...
    def rebuild_index(self) -> Path | None: ...
    def validate(self) -> list[str]: ...
    def sync(self, *, remote_repo: str,
             allow_patterns: tuple[str, ...] | None = None) -> Path: ...
```

要点：方法签名与现有 `MemoryManager` 一一对应；`Toolkit.__init__` 与
`prepare_run_memory` 改为消费 `MemoryBackend`；工厂函数：

```python
def get_memory_backend(name: str, **kwargs) -> MemoryBackend:
    # name: "filesystem"（默认）| "agfs"
```

### 2.2 Episode 数据模型（T2）

```json
{
  "episode_id": "libero10_swap_t2_s0_20261110T031500",
  "cell_tag": "libero10_swap_t2_s0",
  "robot": "libero",
  "suite": "libero_10",
  "regime": "swap",
  "task_id": 2,
  "seed": 0,
  "task_language": "pick up the akita black bowl and ...",
  "outcome": "success",
  "confidence": "verified",
  "memory_ids": ["task-family_libero10_swap_t2"],
  "memory_version": "GPT_6_astra_low",
  "embedding_model": "Qwen3-VL-Embedding-2B@2026.10",
  "created_at": "2026-11-10T03:15:00Z",
  "stats": {"elapsed_s": 41.2, "tool_calls": 17, "output_tokens": 0},
  "assets": {
    "first_frame": "agfs://frames/sha256/ab/cd/abcd....png",
    "success_frame": "agfs://frames/sha256/...",
    "failure_frames": [],
    "seg_overlays": ["agfs://frames/sha256/..."]
  }
}
```

约束：
- `episode.json` 全量落 AGFS `/episodes/<memory_version>/<episode_id>.json`，
  这是事实源；Milvus 只是它的派生索引；
- 帧一律内容寻址、URI 引用，禁止 base64 内联；
- `outcome ∈ {success, failure, aborted}`。

### 2.3 embedding_server（T3）

独立进程，HTTP RPC，风格对齐 `rpent/robots/components/sam3_server.py`。

```
POST /encode
{ "text": "...", "image_uris": ["agfs://...", "file://..."] }
→ { "vector": [float, ...], "model": "Qwen3-VL-Embedding-2B@2026.10", "dim": 1536 }
```

- 文本与图像编码到**同一向量空间**；
- 提供 `--mock` 模式：输出由输入内容 SHA256 播种的确定性伪随机向量，
  维度与真实模型一致（供 CI/无卡环境）；
- 配置项：`--embedding-endpoint`（直连现有服务，优先）、
  `--embedding-model-path`、`--cuda-device`。

### 2.4 Milvus 写路径（T4）

- collection 命名：`episodes__<memory_version>`（非法字符转下划线）；
- partition key：`robot`；
- schema：

| 字段 | 类型 | 说明 |
|---|---|---|
| episode_id | VARCHAR(128), PK | |
| vector | FLOAT_VECTOR(1536) | HNSW（全局检索）；任务族内检索用 FLAT |
| robot / suite / regime | VARCHAR(64) | 标量索引 |
| task_id / seed | INT64 | 标量索引 |
| outcome / confidence | VARCHAR(16) | 标量索引 |
| memory_ids | ARRAY\<VARCHAR(128)\> | 回链知识层 |
| episode_uri | VARCHAR(512) | 指回 AGFS episode.json |

- 开发/CI：`milvus-lite` 本地文件；生产：Standalone；
- upsert 幂等：同一 episode_id 重复入库不产生重复记录。

### 2.5 检索工具（T5）

```python
@tool(readonly=True)
def search_episodes(
    query_text: str | None = None,
    current_frame: str | None = None,   # agfs:// 或 file:// URI
    suite: str | None = None,
    regime: str | None = None,
    outcome: str | None = None,
    top_k: int = 5,
) -> ToolResult:
    """语义检索历史情景记忆。返回 episode 摘要 + 帧 URI + 关联 memory_ids。
    召回后必须用 read_text_file 阅读关联知识叶子正文再决策。"""
```

行为：payload 标量预过滤（robot 恒等于当前机器人）→ ANN → 组装 ToolResult
（含 episode_uri 与 memory_ids）。`query_text` 与 `current_frame` 至少给一个。

提示词更新（`robots/libero/prompts/evaluate.py`，单独 commit）：
把"READ MEMORY"段升级为三段式：
① 读 `MEMORY.md` 索引与任务族叶子路径；
② 调 `search_episodes` 用任务语言 + 当前首帧做语义召回；
③ 对召回结果 `read_text_file` 读正文裁决（保留"登记已读文件"的审计要求）。

### 2.6 mem0 平面（T6）

- 部署：mem0 开源 SDK 独立服务进程 `session_memory_server`（HTTP），
  向量后端配置为 Milvus（collection：`session_mem__prod`）；
- 作用域：`user_id=<操作者ID>`（缺省 `default_operator`），
  `run_id=<cell_tag>`；
- 写入点（仅探索/交互模式）：任务指令提交时、TUI 用户每行输入、
  人工 verdict（success/failure）；
- 读取点：`build_planner` 后，用任务语言 `mem0.search()` 召回偏好，
  注入 `prompt_vars["operator_preferences"]`，prompt 模板渲染时引用；
- 评测模式：mem0 只读（`--memory-profile hf` 或评测套件运行时禁写）。

```python
@tool(readonly=True)
def recall_preferences(topic: str, top_k: int = 5) -> ToolResult:
    """召回当前操作者的偏好与历史确认习惯。"""
```

### 2.7 冻结与回归（T7）

`tools/freeze_memory_version.py`：

1. 语料目录打 tag（HF revision 或本地 git commit 哈希写入 manifest）；
2. Milvus：`episodes__<version>` 置只读（release + 记录到 manifest）；
3. AGFS：`/episodes/<version>/` 与对应 `/corpus/` 前缀切只读挂载；
4. 生成 `memory/<robot>/_internal/freeze/<version>.manifest.json`：

```json
{
  "version": "GPT_6_astra_low",
  "corpus_revision": "hf:RLinf/RPent-memory@<commit>",
  "milvus_collection": "episodes__GPT_6_astra_low",
  "embedding_model": "Qwen3-VL-Embedding-2B@2026.10",
  "frozen_at": "2026-11-10T04:00:00Z"
}
```

回归：同一 suite/task/seed 用冻结语料连跑两次，`diff` 两次 transcript
（剔除时间戳字段后）必须为空。

### 2.8 知识叶子向量索引（T8，可选）

merge 事务发布叶子后，编码 `frontmatter + 正文首段`，upsert 到
`leaves__<version>` collection；`search_episodes` 之外新增
`search_knowledge(query_text, scope?, kind?)`。检索提示词中的 grep 指引
保留为降级路径。

## 3. 写读路径时序

### 写入（任务结束后）

```
main.py 收尾 → backend.merge_memory()（知识层，现有逻辑）
  → 若 solved 或 explore 模式：触发 episode 入库（异步，异常只记日志）：
      1. 从 output_dir 抽关键帧（首帧/成功帧/失败帧/最近 overlay）
      2. 帧 → AGFS /frames/（内容寻址，已存在则复用 URI）
      3. episode.json → AGFS /episodes/<version>/
      4. embedding_server.encode(task_language, first_frame)
      5. Milvus upsert
  → 交互内容另路写 mem0（add，自动对账）
```

### 读取（规划器）

```
规划开始 → 提示词注入 operator_preferences（mem0）
  → 读 MEMORY.md / 任务族叶子（现有）
  → search_episodes(task_language, 当前首帧)（新）
  → read_text_file 关联叶子（现有强制规约）
  → 规划与执行（现有）
```

## 4. 风险与降级

| 风险 | 缓解 |
|---|---|
| AGFS 项目年轻、API 可能变 | 藏在 MemoryBackend 之后，可退回 filesystem |
| embedding 服务不可用 | 检索工具降级为提示词中已有的 grep 路径；episode 仍落 AGFS |
| Milvus 不可用 | 同上；评测链路不依赖向量（向量是增强，不是必须） |
| mem0 开源版功能变动 | 平面 S 与任务记忆物理隔离，摘除不影响主链路 |
| 提示词改动掉点 | T5 提示词单独 commit，git revert 秒级回退 |
