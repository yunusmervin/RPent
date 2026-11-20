# 知识叶子向量索引

知识正文与 YAML frontmatter 仍以 Markdown 为事实源。`leaves__<memory_version>` 是可重建的派生索引，保存 global/task-family 叶子的向量、scope、kind、suite、confidence 和 `leaf_uri`。collection 按版本命名，robot 为 partition key；`leaf_id` 使用 `robot:canonical_id`，避免不同机器人覆盖同名叶子。

## 发布与降级

LIBERO 探索模式在已有 embedding 客户端可用时，为 memory backend 安装懒加载的索引钩子。merge 完成知识叶子的发布和 `MEMORY.md` 更新后，钩子编码原 frontmatter 与正文首段，正文最多取 1000 个字符。新增叶子和 evidence/confidence 更新都会同步；inbox、冲突归档、task-specific 和 results 不进入知识索引。冲突发生时，只索引已发布正文及其更新后的元数据。

embedding 或 Milvus 出错只记录告警，已完成的 Markdown 合并和冲突归档保留。再次重建可恢复派生索引。显式 `confidence: contradicted` 的叶子仍被保留；当 canonical 元数据的 `evidence.contradicted_by` 非空时，派生行的 confidence 也标记为 `contradicted`，语料正文和原 frontmatter 不变。

默认评测不安装写入钩子，也不连接知识索引。冻结语料与 frozen collection 拒绝知识索引写入；继续探索需要另一个可写语料目录。

## 重建索引

以下命令从仓库根目录运行，使用已安装 RPent 的 Python 环境和独立 embedding HTTP 服务。`--memory-dir` 是机器人目录的父目录；与运行任务时直接传入 corpus root 的 `--memory-dir` 含义不同。

```bash
python tools/rebuild_episode_index.py \
  --robot libero --memory-version unversioned --memory-dir memory \
  --collection leaves --embedding-endpoint http://127.0.0.1:8118
```

默认选择 `memory/libero/<memory_version>/`（存在时），否则选择 `memory/libero/`。`--corpus-dir` 可显式选择该机器人目录中的实际语料根目录。索引存入机器人根目录的 `.milvus/episodes.db`，也可用 `--milvus-uri` 指定已有数据库或 Standalone 服务。

`--collection episodes` 保持原有情景索引重建行为，也是默认值；`--collection all` 先重建 episodes，再重建 leaves。leaves 重建只替换当前机器人记录，保留其他机器人分区，且不会因 `contradicted` 状态过滤叶子。目标记录写入成功后，才清理当前机器人的陈旧行。

无模型环境可用 `--mock` 替代 `--embedding-endpoint`，验证重建与查询管线。mock 是由输入哈希决定的伪随机向量，不具备语义能力；查询必须使用同一种 mock 编码器，不能用真实模型查询 mock 建立的索引，也不能据此声称语义效果通过。

## 可选检索

LIBERO 通过 `--enable-knowledge-search` 显式开启工具；默认关闭。以本地语料为例：

```bash
rpent --robot libero --suite libero_10_task --task 2 --seed 0 \
  --memory-profile local --memory-dir memory/libero \
  --enable-knowledge-search --embedding-endpoint http://127.0.0.1:8118 \
  --milvus-uri memory/libero/.milvus/episodes.db
```

本地模式的 memory version 为 `unversioned`。使用 HF 版本时，collection 名称必须匹配所选版本，重建时的 corpus root 必须与本次任务实际选择的 corpus root 一致。`leaf_uri` 是当前机器上的绝对路径，迁移语料目录后应重建索引。CLI 与 Dashboard 都保留该开关；当前接线面向 LIBERO。

工具签名为 `search_knowledge(query_text, scope=None, kind=None, top_k=5)`。scope 可选 `global` 或 `task-family`；kind 可选现有知识类型；top_k 为 1–100 的整数。检索固定按当前 robot 预过滤，返回叶子标量信息、相似度及 `leaf_uri`。它不会自动读取正文，也不会将结果登记为已读；规划器仍需用 `read_text_file` 阅读对应正文后再决策。

缺少索引或初始化连接失败时不注册工具；工具注册后，检索执行中若 embedding 或 Milvus 不可用，会返回结构化错误与 `MEMORY.md`、global/task-family 的 grep 降级路径。本任务不修改提示词，不改变现有 grep 指引。

## 验收范围

```bash
pytest tests/unit_tests/ -q
pytest tests/unit_tests/episode/test_knowledge_index.py -q
pytest tests/unit_tests/ -k "memory or episode" -q
ruff check .
```

离线测试使用真实 Milvus Lite 和受控编码器，验证三条措辞不同的 global 叶子可按人话查询返回对应的 top-1 结果，以及冲突稿排除、索引失败降级、机器人隔离、否定证据保留和重建一致性。真实 Qwen3-VL-Embedding-2B 的语义效果需要连接已部署的服务另行验证。

实现入口：[发布钩子](../rpent/memory/manager.py)、[知识索引](../rpent/memory/knowledge.py)、[检索工具](../rpent/memory/knowledge_search.py)、[重建工具](../tools/rebuild_episode_index.py)。
