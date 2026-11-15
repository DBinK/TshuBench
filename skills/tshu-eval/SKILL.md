---
name: tshu-eval
description: TshuBench（天书榜）大模型评测与社区贡献指南。仅在用户显式要求执行天书榜评测、或通过提示词/斜杠命令调用时使用；不要从一般开发或代码对话上下文自动触发。
license: MIT
compatibility: 需要 uv/uvx 与 Python 3.10+
metadata:
  author: DBinK
  version: "0.2.0"
---

# TshuBench · 天书榜评测技能

## 什么是 TshuBench（天书榜）

**TshuBench（天书榜）** 是一个专门量化中文大模型工程回答中「不说人话」与结构膨胀的确定性 AST 评测基准（无 LLM 裁判，Zero-LLM）：
- **六维病灶**：长链推演（CCI）、排版膨胀（FMT）、报菜名（ENU）、套话模板（TIC）、决断推延（CVD）、防御叠甲（HED）。
- **评分机制**：共 36 道工程场景题，采用反向扣分制，分数越低越讲人话（0 分为未检出病灶）。
- **准入与排位**：仅收录在 [models.dev](https://models.dev) 可查到公开元数据的模型；同模型若有多个思考档位，主榜取天书指数最低的档位上榜，展开可看全档位对比。

## 执行流程

1. **确认目标模型与思考档位**
   - 确认用户要评测的模型 ID（须能在 `models.dev` 查到公开元数据）及思考档位（`minimal` / `low` / `medium` / `high` / `none`）。

2. **检查 API 凭证（缺失则主动询问）**
   - 检查环境变量中是否已存在对应的 API Key（如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`OPENROUTER_API_KEY` 等）与接口地址。
   - **重要**：如果环境变量中没有找到可用密钥，必须主动向用户询问 API Key 与 Base URL，拿到后再继续，不得直接报错退出。

3. **运行 36 题全量评测**
   - 若在仓库外运行，使用 `uvx` 免安装执行（内置 36 题数据集）：
     ```bash
     uvx --refresh --from git+https://github.com/DBinK/TshuBench tshu run \
       --model "<provider/model>" \
       --base-url "<base_url>" \
       --api-key-env "<ENV_VAR_NAME>" \
       --out "runs/<provider>/<model_slug>_<effort>.jsonl"
     ```
   - 若已在 TshuBench 仓库目录内，改用 `uv run tshu run ...`。
   - 目录与命名规范：`runs/<provider>/<model_slug>_<effort>.jsonl`（全部小写，斜杠替换为短横线）。

4. **本地完整性校验与试算**
   - 运行自检命令：
     ```bash
     uvx --refresh --from git+https://github.com/DBinK/TshuBench tshu score \
       "runs/<provider>/<model_slug>_<effort>.jsonl" \
       --out /tmp/tshu_verify
     ```
   - 确认输出报告显示 `样本 36`，无缺题、无截断报错，且未修改既有数据。

5. **汇报结果与引导提交 PR**
   - 向用户汇报该模型的天书指数、95% 置信区间、所属梯队与六维扣分概况。
   - 引导用户向 `DBinK/TshuBench` 提交 Pull Request（仅新增 `runs/<provider>/<model_slug>_<effort>.jsonl`），CI 会自动校验并在 PR 评论区回帖战报预览。
