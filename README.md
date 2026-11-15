# TshuBench · 天书榜

> **量化大模型中文工程回答中的表达负担与结构膨胀**  
> 基于确定性 AST 语法树与词法规则，无 LLM 裁判（Zero-LLM），100% 可复现。

**在线交互榜单**：[https://tshu.dbink.top/](https://tshu.dbink.top/)

---

## 简介

**TshuBench（天书榜）** 从真实工程研发场景中提炼了 36 道高频测试题（涵盖排障、代码审查、技术澄清与发版交付），用于度量模型输出中的六类典型文风病灶：

| 维度 | 说明 | 权重 |
| :--- | :--- | :---: |
| **长链推演（CCI）** | 正文夹杂改口、假设推导与长链嵌套从句 | 24% |
| **报菜名（ENU）** | 单位字数内高密度顿号、斜杠与行内代码堆砌 | 20% |
| **排版膨胀（FMT）** | 简单问题强行使用多级标题、嵌套列表与大表格 | 16% |
| **套话模板（TIC）** | 固定公文腔、总结话术与机械过渡句 | 16% |
| **决断推延（CVD）** | 首句回避核心结论，先做冗长背景复述 | 12% |
| **防御叠甲（HED）** | 过度使用免责声明与模糊限定词 | 12% |

本榜采用**反向扣分制**：分数越低越接近自然简洁的工程表达，0 分表示未检出预设病灶（仅评估表达风格，不判定技术结论正误）。同一模型若存在多个思考档位，主榜取天书指数最低的档位上榜，点击行可展开查看全档位对比与典型高扣分样本。

> [!NOTE]
> 本榜单定位为工程趣味性观测基准，不构成严谨学术研究成果，模型实际表现请以真实使用体验为准。欢迎通过 Issue / PR 贡献新的文风指标与评测算法，下一代榜单正在规划与制作中。

---

## 快速上手

### 方式一：AI Agent 自动化评测（推荐）

将以下提示词直接复制发送给您的 AI 编程助手（如 Claude Code、Cursor、Cline、Antigravity 等）：

```markdown
请帮我评测模型 [请在此填写模型 ID，如 deepseek/deepseek-v4.1-flash]，并准备提交到 TshuBench（天书榜）：

1. 运行 `npx skills add DBinK/TshuBench` 安装 `tshu-eval` 技能。
2. 加载 `tshu-eval` 技能并严格按照其中的规范与步骤执行评测、校验与 PR 准备。
```

### 方式二：手动命令行运行

本地安装 [uv](https://docs.astral.sh/uv/) 后，无需克隆仓库即可通过 `uvx` 直接运行（已内置 36 题标准测试集）：

```bash
# 1. 运行指定模型的 36 题全量评测
uvx --refresh --from git+https://github.com/DBinK/TshuBench tshu run \
  --model <provider/model-id> \
  --base-url <API_BASE_URL> \
  --api-key-env <API_KEY_ENV_VAR> \
  --out runs/<provider>/<model_slug>_<effort>.jsonl

# 2. 对样本执行确定性打分并生成本地战报
uvx --refresh --from git+https://github.com/DBinK/TshuBench tshu score \
  runs/<provider>/<model_slug>_<effort>.jsonl \
  --out reports \
  --html web/index.html
```

若已在本地克隆仓库，可使用 `uv sync` 安装开发依赖后直接执行 `uv run tshu ...` 与 `uv run pytest`。

---

## 社区贡献

欢迎提交在 [models.dev](https://models.dev) 可查到公开元数据的模型评测结果。请将完整 36 题样本文件放置于 `runs/<provider>/<model_slug>_<effort>.jsonl` 并提交 Pull Request，详细准入规则与自动化门禁说明请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 扩展文档

- [docs/methodology.md](docs/methodology.md)：六大文风病灶定义、AST 语法树判定算法、Schema 与加权体系。
- [docs/findings.md](docs/findings.md)：17 款主流模型实测分析、分半信度检验与方法论局限。
- [docs/task-sources.md](docs/task-sources.md)：独立盲测集构建标准与题库扩增规划。
