# 冻结任务记忆并检查双跑一致性

冻结把指定机器人的知识语料、情景向量 collection 和 AGFS 情景目录绑定到同一版本。完成后可以检查快照是否完整，再用相同 suite、task 和 seed 连续评测两次，比较任务 transcript。

Markdown 与 JSON/JSONL 语料仍是事实源，Milvus 是可重建的派生索引。接口契约见[记忆设计](memory-redesign.md#27-冻结与回归t7)，实现入口是[冻结工具](../tools/freeze_memory_version.py)、[检查工具](../tools/check_freeze.py)和[双跑脚本](../tools/replay_consistency.sh)。

## 准备语料和服务

从仓库根目录运行以下命令，使用已安装 RPent 和 episode 依赖的 Python 环境。`--memory-dir` 是机器人目录的父目录：例如 `memory/libero/` 对应 `--memory-dir memory --robot libero`。Milvus 中必须已有该版本的 collection，名称为 `episodes__<version>`，其中非法字符替换为下划线。

冻结前停止向该版本写入的探索任务和其他写进程。默认选择 `memory/<robot>/<version>/`（存在时），否则选择 `memory/<robot>/`；可通过 `--corpus-dir` 指定该机器人目录内的知识语料目录。本地 Git 来源要求语料文件已提交，且内容与记录的 commit 一致。HF 来源使用 `--memory-repo owner/repo@revision` 指定仓库，冻结工具解析并校验不可变 revision。

AGFS LocalFS 没有原生只读配置项。RPent 后端持久化只读策略，并去掉语料与情景目录的写权限；生成的原生挂载配置指向这些目录。使用外接 AGFS 服务时，应在冻结后重启服务，确认它挂载相同的本地目录，且服务用户受这些文件权限约束。操作系统的特权用户仍可改变文件权限；后续检查会核对文件内容和冻结元数据。

## 冻结并检查版本

下面以本地版本 `v1` 为例。先确认语料已经提交并已生成对应的 Milvus collection，再执行：

```bash
python tools/freeze_memory_version.py \
  --robot libero --version v1 \
  --memory-dir memory \
  --milvus-uri memory/libero/.milvus/episodes.db

python tools/check_freeze.py \
  --robot libero --version v1 \
  --memory-dir memory \
  --milvus-uri memory/libero/.milvus/episodes.db
```

冻结会记录语料 revision、embedding model、Milvus collection、冻结时间、绝对语料路径和内容摘要，并生成 `memory/libero/_internal/freeze/v1.manifest.json`。它核对语料的完整文件清单与来源修订，确认当前机器人的向量记录与 canonical episode 的标量信息一致，并保存包含向量的行摘要。随后设置 collection 的 `frozen=true` 和 `frozen_at` properties，release collection，并锁定语料与对应情景目录。语料正文保持原样。本地 Milvus 数据库路径必须位于该机器人目录内；Standalone 使用服务 URI。

检查工具返回 `0` 表示 manifest 与实际状态一致，返回 `1` 表示状态缺失、不一致或服务不可用，参数错误返回 `2`。按错误信息检查语料路径与内容、Milvus collection properties 与行摘要、AGFS 只读配置与目录权限。检查不会重新访问 HF 或调用 embedding 服务；Milvus 核验会临时 load collection，查询后再次 release。冻结版本的向量写路径抛出 `RuntimeError`，显式 episode 记录也会在提取帧之前被拒绝；需要继续探索时，使用另一个版本和可写存储目录。

## 连续评测两次

实际评测需要对应机器人的模拟器、模型服务和规划器已准备就绪。选用新的输出目录；脚本拒绝复用已有目录。下面的任务两次读取同一份冻结语料，并使用相同的命令参数：

```bash
RPENT_PYTHON="$(command -v python)" tools/replay_consistency.sh \
  --robot libero --version v1 \
  --memory-dir memory \
  --suite libero_10_task --task 2 --seed 0 \
  --output-dir /tmp/rpent-v1-replay \
  -- --planner api --model openai:gpt-6-astra
```

脚本默认启动 `rpent`，从 manifest 取得冻结语料路径与 Milvus URI，并固定传入 `--memory-profile local`。这里的 `local` 选择已冻结的本地快照；未传 `--explore`，因此仍是只读评测。脚本拒绝覆盖任务身份、语料来源、输出目录、backend、探索和 episode 写入等参数。`--` 后可以传入规划器、模型和服务连接等任务参数。

每次运行前后均执行冻结检查，并确认 manifest 未被替换。为使 transcript 内的输出路径一致，两次任务都使用输出目录下的 `work/`，运行结束后分别保存为 `run-1/` 和 `run-2/`。原始 transcript 和其他任务产物保存在这两个目录，规范化 transcript 另存为 `run-1.normalized.json` 和 `run-2.normalized.json`。

比较保持消息顺序、工具调用、结果、计数、标识符和路径，仅递归删除以下时序字段：`timestamp`、`created_at`、`updated_at`、`started_at`、`finished_at`、`frozen_at`、`elapsed_s`、`agent_elapsed_s`、`duration_s`、`duration_ms`。CLI 追加写入的多个 JSON 对象也会逐条解析。API 规划器把工具结果存为 JSON 字符串，脚本仅对 `role=tool` 且内容是完整 JSON 对象或列表的消息解码，再按相同规则剔除时序字段。普通文本、截断的工具输出、随机标识符和模型输出差异会保留。

成功时输出 `Replay consistent: normalized transcript diff is empty.`，退出码为 `0`。不一致时退出码为 `1`，并打印首个分歧的 JSON 路径，例如 `$["transcript_10_task_t2_s0.json"][0]["messages"][3]["content"]`。用下面的命令查看完整差异：

```bash
diff -u /tmp/rpent-v1-replay/run-1.normalized.json \
  /tmp/rpent-v1-replay/run-2.normalized.json
```

冻结检查失败、任务进程失败、缺少 transcript 或 manifest 在运行中改变时，脚本也返回非零并停止。固定记忆与 seed 不会消除模型采样和外部服务的随机性；出现差异时应检查首处分歧及两次原始任务产物。相同 transcript 本身不代表任务成功，任务成功需要查看机器人环境的原生判定。

## 离线检查脚本控制流

没有模拟器时，可以使用自定义可执行 runner 检查两次调用、冻结检查和 transcript 比较。这个检查仍需要一份可通过检查工具验证的冻结快照。以下 runner 只写入一个明确标记为离线检查的 transcript：

```bash
cat > /tmp/rpent-offline-replay-runner <<'PY'
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
for option in ("output-dir", "suite", "task", "seed", "memory-profile", "memory-dir"):
    parser.add_argument("--" + option, required=True)
args, _ = parser.parse_known_args()
output = Path(args.output_dir)
output.mkdir(parents=True)
record = {
    "offline_smoke": True,
    "suite": args.suite,
    "task": int(args.task),
    "seed": int(args.seed),
    "environment_success": None,
    "timestamp": time.time(),
    "messages": [{"role": "assistant", "content": "offline runner invoked"}],
}
(output / "transcript_offline.json").write_text(json.dumps(record))
PY
chmod +x /tmp/rpent-offline-replay-runner

RPENT_PYTHON="$(command -v python)" tools/replay_consistency.sh \
  --robot libero --version v1 --memory-dir memory \
  --suite libero_10_task --task 2 --seed 0 \
  --output-dir /tmp/rpent-v1-offline-replay \
  --runner /tmp/rpent-offline-replay-runner
```

预期结果为两次 runner 调用、四次冻结检查通过、规范化 diff 为空。该结果只验证回归脚本的控制流；模拟器执行、动作正确性和实际任务一致性需要单独运行上面的真实评测命令。
