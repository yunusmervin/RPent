# AGENTS.md — 本仓库的智能体开发准则

本文件约束一切编程智能体（含 Kimi Code）在本仓库中的行为。**任何任务卡与本文件冲突时，以本文件为准。**

## 架构事实（不可违反）

1. **Markdown 语料库是唯一事实源（canonical source of truth）**。
   Milvus、mem0、AGFS 向量挂载点全部是**派生索引**，必须能随时从
   `memory/` 下的 Markdown + JSON/JSONL 语料完整重建。
2. **评测模式（未传 `--explore`）下，一切记忆写路径必须关闭**：
   语料库、episode 向量、mem0 全部只读。这是 LIBERO-PRO 榜单可复现性的红线。
3. `rpent/cli/` 不允许被任何其他 `rpent` 模块 import（防循环依赖）。
4. 服务化原则：模型推理（VLA/SAM3/embedding）一律跑在独立进程，通过
   HTTP RPC 通信，主进程内**禁止** import 模型权重代码。
5. 记忆按机器人隔离：任何路径访问不得越过 `memory/<robot>/` 边界。

## 必守流程

1. 动手前先读：`docs/memory-redesign.md` 与对应 `tasks/T<n>.md`。
2. 改任何记忆行为前，先跑基线确认全绿：
   ```bash
   pytest tests/unit_tests/ -q
   ```
3. 完成后必须跑验收命令并在总结中**逐条报告**结果（通过/失败 + 关键输出摘要）：
   ```bash
   pytest tests/unit_tests/ -q
   pytest tests/unit_tests/ -k "memory or episode" -q
   ruff check .
   ```
4. 一个任务卡 = 一个分支 = 一个 PR。不夹带任务卡之外的改动。
5. 提示词文件（`robots/*/prompts/`）的任何改动必须单独成一个 commit。

## 目录地图

| 路径 | 作用 |
|---|---|
| `rpent/memory/` | 记忆核心：manager（合并/索引/同步）、loading、source、model |
| `rpent/tools/` | 工具基类（base.py）、工具集（toolkit.py）、通用工具（common.py） |
| `rpent/planner/` | 规划器：base / claude_code / codex / api_loop / flash |
| `rpent/robots/` | 机器人框架基类与共享组件（components/） |
| `robots/<name>/` | 机器人插件：robot_spec / tools / prompts / memory / flash |
| `tests/unit_tests/` | 单元测试，含 memory contract 测试 |
| `docs/memory-redesign.md` | 多模态记忆重构总体设计（本项目的单一事实来源） |
| `tasks/` | T0–T8 任务卡 |

## 禁止事项

- 不引入 AGPL/SSPL 等传染性协议依赖（OpenViking、MinIO 均在禁止之列；
  对象存储用 SeaweedFS 或文件系统内容寻址）。
- 不改动 `robots/*/prompts/` 下提示词（除非任务卡明确点名且单独成 commit）。
- 不在评测路径上新增网络调用。
- 不修改 `tests/unit_tests/` 中既有测试的断言（发现测试错误时，先提 issue 说明，
  由人类决定；允许新增测试）。
- 不把 base64 图像等大对象写入日志或 prompt 变量。

## 技术栈锁定

| 组件 | 选型 | 说明 |
|---|---|---|
| 向量库 | Milvus（开发用 milvus-lite，生产 Standalone） | collection-per-version，partition_key=robot |
| 多模态编码 | Qwen3-VL-Embedding-2B（独立服务） | 版本钉死，写进语料元数据 |
| 统一命名空间 | AGFS（LocalFS/S3FS + 自研 MMVectorFS） | Apache-2.0 |
| 会话/偏好记忆 | mem0 开源 SDK | 向量后端指向 Milvus |
| 帧存储 | AGFS /frames（SHA256 内容寻址） | 禁止 base64 内联进 JSON |
