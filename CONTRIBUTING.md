# 贡献指南（CONTRIBUTING）

欢迎向 **TshuBench · 天书榜** 贡献公开模型的评测结果！为保证榜单的公平性、可复现性与防刷分机制，所有社区提交的 Pull Request 均需遵守以下规范。

## 1. 准入标准

1. **models.dev 元数据可查**：仅接受在 [models.dev](https://models.dev) 中能够查到公开元数据的模型（官方发布、规范命名、可公开复现）。不接受私人微调模型、未公开权重的内部测试路由、或掺杂未知系统提示词的第三方套壳代理。
2. **完整 36 题覆盖**：每次提交的 `.jsonl` 样本文件必须完整覆盖 `eval_suite_v2.jsonl` 中的全部 36 道标准题目，不允许挑选子集、缺题或重复题目。
3. **原始回复不可篡改**：模型原始输出须直接通过 `tshu run` 生成并原样保留，不得人工删改文本。

## 2. 目录与命名规范

请将评测样本文件放置在 `runs/<provider>/` 目录下，命名格式如下：

```text
runs/<provider>/<model_slug>_<effort>.jsonl
```

- `<provider>`：模型厂商或标准聚合路由名（如 `anthropic`、`openai`、`google`、`deepseek`、`qwen` 等，全小写）。
- `<model_slug>`：模型标识，斜杠或特殊字符替换为短横线 `-`（如 `deepseek-v4.1-flash`）。
- `<effort>`：思考档位（如 `minimal`、`low`、`medium`、`high`、`none`）。

示例：
- `runs/deepseek/deepseek-v4.1-flash_low.jsonl`
- `runs/google/gemini-4-argon_medium.jsonl`

## 3. 多思考档位归并规则

同一模型允许提交多个不同思考档位（`reasoning_effort`）的评测结果：
- **主榜排位**：系统自动取该模型所有已提交档位中**天书指数最低（最讲人话）**的档位作为代表行参与主榜排位。
- **抽屉下钻**：点击主榜任意模型行展开详情后，可查看该模型所有已提交思考档位的横向对比数据。

## 4. 快速评测与提交流程

### 方式一：使用 `uvx` 一键运行

无需克隆仓库，本地安装 `uv` 后即可直接运行：

```bash
# 1. 运行评测（自动内置 36 题数据集）
uvx --from git+https://github.com/DBinK/TshuBench tshu run \
  --model <provider/model-id> \
  --base-url <API_BASE_URL> \
  --api-key-env <YOUR_API_KEY_ENV> \
  --out runs/<provider>/<model_slug>_<effort>.jsonl

# 2. 本地自检打分
uvx --from git+https://github.com/DBinK/TshuBench tshu score \
  runs/<provider>/<model_slug>_<effort>.jsonl \
  --out /tmp/tshu_check
```

### 方式二：交给您的 AI Agent 自动完成

直接将 [README 快速上手](README.md#方式一ai-agent-自动化评测推荐) 中的提示词复制给您的 AI 编程助手（如 Claude Code、Cursor、Cline、Antigravity 等），它会自动安装 `tshu-eval` 技能并完成 API 密钥确认、评测执行、完整性校验与 PR 准备。

## 5. 自动化门禁（CI）

提交 PR 后，GitHub Actions 将自动执行以下检查：
- 校验 PR 仅在 `runs/` 下新增 `.jsonl` 文件，禁止修改或删除已有样本，且未篡改核心源码与题库。
- 校验 Pydantic Schema 与 36 道题号的 100% 精确匹配。
- 自动运行 `tshu score` 并在 PR 评论区回复该模型的诊断战报预览。
- PR 合并进入 `main` 分支后，GitHub Pages 将自动重新汇算全榜数据并发布上线。
