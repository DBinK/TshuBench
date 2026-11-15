# AI Agent 一键评测提示词

将以下提示词直接复制发送给您的 AI 编程助手（支持 Claude Code、Cursor、Cline、Roo Code、Antigravity 等），即可全自动完成模型评测与 PR 准备：

```markdown
请帮我评测模型 [请在此填写模型 ID，如 deepseek/deepseek-v3]，并准备提交到 TshuBench（天书榜）：

1. **检查与获取凭证**：
   - 先检查当前环境变量中是否已配置所需的 API Key（如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`OPENROUTER_API_KEY` 等）与接口地址。
   - 如果环境变量中没有找到可用的 API Key，请**主动向我询问** API Key 与 Base URL，拿到后再继续，不要直接报错退出。

2. **执行 36 题全量评测**：
   - 使用 `uvx --from git+https://github.com/DBinK/TshuBench tshu run`（若当前已在 TshuBench 仓库目录内，则用 `uv run tshu run`）执行评测。
   - 输出路径严格遵循：`runs/<provider>/<model_slug>_<effort>.jsonl`。

3. **本地校验与试算**：
   - 运行 `tshu score <输出文件路径> --out /tmp/tshu_verify`。
   - 确认输出样本数必须严格等于 `36`（无缺题、无截断报错）。

4. **整理结果**：
   - 评测通过后，向我汇报该模型的天书指数、95% 置信区间及六维扣分概况，并引导我提交 Pull Request 到 `DBinK/TshuBench`。
```
