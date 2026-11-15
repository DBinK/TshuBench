---
name: tshu-eval
description: 使用 tshu CLI 对指定大模型执行 TshuBench 天书榜 36 题评测，完成本地完整性校验并生成符合社区贡献规范的 JSONL 文件。
---

# TshuBench 自动化评测技能

## 执行流程

1. **确认目标模型与思考档位**
   - 确认用户要评测的模型 ID（须能在 [models.dev](https://models.dev) 查到公开元数据）与思考档位（`minimal` / `low` / `medium` / `high`）。

2. **检查 API 凭证（缺失则主动询问）**
   - 检查环境变量中是否已存在对应的 API Key 与 Base URL。
   - **重要**：若环境变量中未配置相关密钥，必须主动向用户询问 API Key 与接口地址，获得确认后再执行命令。

3. **执行评测**
   - 在仓库外使用：
     ```bash
     uvx --from git+https://github.com/DBinK/TshuBench tshu run \
       --model "<provider/model>" \
       --base-url "<base_url>" \
       --api-key-env "<ENV_VAR_NAME>" \
       --out "runs/<provider>/<model_slug>_<effort>.jsonl"
     ```
   - 在仓库内使用 `uv run tshu run ...`。

4. **自查完整性与试算**
   - 执行 `tshu score runs/<provider>/<model_slug>_<effort>.jsonl --out /tmp/tshu_verify`。
   - 确认输出报告显示 `样本 36`，无缺失题目。

5. **汇报与提交指引**
   - 向用户汇报天书指数、所属梯队与六维扣分，并提示用户可提交 PR 至 `runs/` 目录。
