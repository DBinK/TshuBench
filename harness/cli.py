"""TshuBench 统一 CLI 入口（基于 Typer）。

用法示例：
    uv run python -m harness.cli score runs/real/*.jsonl
    uv run python -m harness.cli run --demo --out-prefix runs/demo
    uv run python -m harness.cli suite --models deepseek-chat gpt-5 --out-dir runs/real
"""

from pathlib import Path

import typer

from harness.audit import compute_audit_report, print_audit_report
from harness.run import execute_models, run_demo
from harness.score import load_tasks, run_scoring

app = typer.Typer(help="TshuBench · 天书榜评测与打分命令行工具", no_args_is_help=True)


@app.command("run")
def run_cmd(
    tasks: Path = typer.Option(Path("tasks/eval_suite_v2.jsonl"), help="题目集 JSONL 路径"),
    models: list[str] = typer.Option([], "--model", "-m", help="模型名，可传多次，支持 model:effort"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", help="OpenAI 兼容接口地址"),
    api_key_env: str = typer.Option("OPENAI_API_KEY", help="API Key 环境变量名"),
    out_dir: Path = typer.Option(Path("runs"), help="输出目录"),
    concurrency: int = typer.Option(8, help="并发请求数"),
    temperature: float = typer.Option(0.3, help="采样温度"),
    max_tokens: int = typer.Option(8192, help="单次最大生成 token 数"),
    reasoning_effort: str = typer.Option("", help="默认思考档位 low/medium/high"),
    timeout: float = typer.Option(180.0, help="请求超时时间（秒）"),
    limit: int = typer.Option(0, help="仅测前 N 题，0 表示全量"),
    demo: bool = typer.Option(False, help="生成离线 demo_tidy/demo_slop 样本"),
    out_prefix: str = typer.Option("runs/demo", help="demo 模式输出文件前缀"),
) -> None:
    """采样模型回复，或生成离线演示样本。"""
    if demo:
        loaded = load_tasks(tasks)
        run_demo(loaded, out_prefix)
        return
    if not models:
        raise typer.BadParameter("非 demo 模式至少需要传一次 --model / -m")
    execute_models(
        tasks,
        models=models,
        base_url=base_url,
        api_key_env=api_key_env,
        out_dir=out_dir,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        limit=limit,
    )


@app.command("score")
def score_cmd(
    runs: list[Path] = typer.Argument(..., help="待打分的模型输出 JSONL 文件列表"),
    tasks: Path = typer.Option(Path("tasks/eval_suite_v2.jsonl"), help="题目集 JSONL 路径"),
    out: Path = typer.Option(Path("reports"), help="报告输出目录"),
    html: Path = typer.Option(Path("web/index.html"), help="静态网页榜单输出路径"),
) -> None:
    """对模型输出 JSONL 执行确定性打分并生成报告。"""
    run_scoring(tasks, list(runs), out, html)


@app.command("audit")
def audit_cmd(
    runs: list[Path] = typer.Argument(..., help="待体检的模型输出 JSONL 文件列表"),
    tasks: Path = typer.Option(Path("tasks/eval_suite_v2.jsonl"), help="题目集 JSONL 路径"),
    n_perm: int = typer.Option(10000, help="配对符号置换检验次数"),
) -> None:
    """运行零人工成本统计体检（基线对照、方差贡献、PCA 与配对置换检验）。"""
    report = compute_audit_report(tasks, list(runs), n_perm=n_perm)
    print_audit_report(report)


@app.command("suite")
def suite_cmd(
    models: list[str] = typer.Option(..., "--model", "-m", help="模型名，可传多次，支持 model:effort"),
    tasks: Path = typer.Option(Path("tasks/eval_suite_v2.jsonl"), help="题目集 JSONL 路径"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", help="OpenAI 兼容接口地址"),
    api_key_env: str = typer.Option("OPENAI_API_KEY", help="API Key 环境变量名"),
    out_dir: Path = typer.Option(Path("runs"), help="采样结果目录"),
    reports_dir: Path = typer.Option(Path("reports"), help="报告输出目录"),
    html: Path = typer.Option(Path("web/index.html"), help="静态网页榜单输出路径"),
    concurrency: int = typer.Option(8, help="并发请求数"),
    temperature: float = typer.Option(0.3, help="采样温度"),
    max_tokens: int = typer.Option(8192, help="单次最大生成 token 数"),
    reasoning_effort: str = typer.Option("", help="默认思考档位 low/medium/high"),
    timeout: float = typer.Option(180.0, help="请求超时时间（秒）"),
    limit: int = typer.Option(0, help="仅测前 N 题，0 表示全量"),
) -> None:
    """串联采样模型并立即生成最新榜单与战报。"""
    written = execute_models(
        tasks,
        models=models,
        base_url=base_url,
        api_key_env=api_key_env,
        out_dir=out_dir,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        limit=limit,
    )
    run_scoring(tasks, list(written), reports_dir, html)


if __name__ == "__main__":
    app()
