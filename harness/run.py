"""TshuBench 调用器：采样 OpenAI 兼容接口，或生成 demo 合成数据。

真实模型：
    uv run python -m harness.run --model deepseek-chat \
        --base-url https://api.deepseek.com/v1 --out runs/deepseek-chat.jsonl

离线演示：
    uv run python -m harness.run --demo --out-prefix runs/demo
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
from loguru import logger

from harness.models import Sample, Task
from harness.score import load_tasks
from harness.text import is_truncated

THINK_BLOCK = re.compile(r"<(think|thinking|reasoning|analysis|thought)>.*?</\1>", re.S | re.I)


def strip_think(text: str) -> tuple[str, int]:
    chars = sum(len(m.group(0)) for m in THINK_BLOCK.finditer(text))
    return THINK_BLOCK.sub("", text).strip(), chars


async def call_once(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    task: Task,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str = "",
) -> Sample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            payload["max_tokens"] = max_tokens * (2**attempt)
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            text, inline_chars = strip_think(content)
            if not text:
                last_error = RuntimeError(f"空回复，finish_reason={choice.get('finish_reason')}")
                continue
            if choice.get("finish_reason") == "length" and is_truncated(text) and attempt < 2:
                last_error = RuntimeError("正文被预算截断，加倍预算重试")
                continue
            return Sample(
                task_id=task.id,
                model=model,
                response=text,
                think_chars=len(reasoning) + inline_chars,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                thinking=reasoning,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001 - 网络重试需要兜底
            last_error = exc
            await asyncio.sleep(2**attempt)
    logger.error(f"{task.id} 调用失败：{last_error}")
    return Sample(task_id=task.id, model=model, response="", reasoning_effort=reasoning_effort)


def tidy_response(task: Task) -> str:
    anchors = task.answer_anchors
    entities = task.actionable_entities
    first = anchors[0] if anchors else (entities[0] if entities else "按现有配置处理")
    second = anchors[1] if len(anchors) > 1 else (entities[1] if len(entities) > 1 else "")
    context = "、".join(entities[:3])
    if second:
        return f"{first}。关键在 {second}，涉及 {context}。"
    return f"{first}。涉及 {context}。"


SLOP_TEMPLATE = (
    "这是一个非常好的问题！你说得对，这个问题确实需要认真对待。\n\n"
    "### 一、背景分析\n\n"
    "在当前的工程实践中，关于这个问题我们需要从底层逻辑出发，进行多维度的系统性分析与全面的评估工作，"
    "并且需要结合业务场景、团队能力、历史包袱以及未来的可扩展性进行综合性的判断与权衡。\n\n"
    "### 二、方案权衡\n\n"
    "一方面，前者具备明显的优势，另一方面，后者也有其不可替代的价值，两者各有优劣，具体取决于你的实际情况。\n"
    "换句话说，没有绝对正确的答案，需要结合上下文权衡。\n"
    "如果下游服务持续超时并且一旦线程池已经打满同时在高峰期的情况下我们需要立即评估是否需要熔断降级以及如何设计恢复策略并且还要考虑对上游业务造成的潜在影响和后续的容量规划。\n"
    "不过话说回来，值得注意的是一些边界条件，从另一个角度来看，还需要考虑团队的心智负担与长期维护成本。\n\n"
    "### 三、总结\n\n"
    "综上所述，{a0} 与 {a1} 各有优劣，建议结合具体业务场景综合评估，必要时可以先做一个小范围试点，再逐步推广。"
    "以上内容仅供参考，具体请结合实际情况判断。\n\n"
    "希望以上分析对你有所帮助，祝你的项目顺利推进！"
)


def slop_response(task: Task) -> str:
    anchors = task.answer_anchors or task.actionable_entities or ["方案 A", "方案 B"]
    first = anchors[0]
    second = anchors[1] if len(anchors) > 1 else "备选方案"
    return SLOP_TEMPLATE.format(a0=first, a1=second)


def write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "task_id": s.task_id,
                "model": s.model,
                "response": s.response,
                "think_chars": s.think_chars,
                "latency_ms": s.latency_ms,
                "thinking": s.thinking,
                "reasoning_effort": s.reasoning_effort,
            },
            ensure_ascii=False,
        )
        for s in samples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.success(f"写出 {len(samples)} 条 -> {path}")


def run_demo(tasks: dict[str, Task], prefix: str) -> None:
    tidy = [Sample(task_id=t, model="demo-tidy", response=tidy_response(task)) for t, task in tasks.items()]
    slop = [Sample(task_id=t, model="demo-slop", response=slop_response(task)) for t, task in tasks.items()]
    base = Path(prefix)
    write_jsonl(base.with_name(base.name + "_tidy.jsonl"), tidy)
    write_jsonl(base.with_name(base.name + "_slop.jsonl"), slop)


async def execute_live(
    tasks: dict[str, Task],
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    out: str | Path,
    concurrency: int = 8,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    reasoning_effort: str = "",
    timeout: float = 180.0,
    limit: int = 0,
) -> None:
    """并发采样指定模型并将结果写入 JSONL。"""
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise SystemExit(f"环境变量 {api_key_env} 为空")
    semaphore = asyncio.Semaphore(concurrency)
    selected = list(tasks.values())[:limit] if limit else list(tasks.values())

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def worker(task: Task) -> Sample:
            async with semaphore:
                return await call_once(
                    client,
                    base_url,
                    api_key,
                    model,
                    task,
                    temperature,
                    max_tokens,
                    reasoning_effort,
                )

        samples = await asyncio.gather(*(worker(t) for t in selected))
    write_jsonl(Path(out), list(samples))


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def execute_models(
    tasks_path: str | Path,
    *,
    models: list[str],
    base_url: str,
    api_key_env: str,
    out_dir: str | Path,
    concurrency: int = 8,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    reasoning_effort: str = "",
    timeout: float = 180.0,
    limit: int = 0,
) -> list[Path]:
    """按模型配置顺序完成采样，返回写出的 JSONL 路径列表。"""
    tasks = load_tasks(tasks_path)
    logger.info(f"载入 {len(tasks)} 道题")
    out_base = Path(out_dir)
    written: list[Path] = []
    for spec in models:
        model, _, effort = spec.partition(":")
        eff = effort or reasoning_effort
        target = out_base / f"{safe_name(model)}.jsonl"
        logger.info(f"开始 {model}，思考档位 {eff or '默认'}")
        asyncio.run(
            execute_live(
                tasks,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                out=target,
                concurrency=concurrency,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=eff,
                timeout=timeout,
                limit=limit,
            )
        )
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="TshuBench 调用器")
    parser.add_argument("--tasks", default="tasks/eval_suite_v2.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--models",
        nargs="+",
        help="批量模型，可写 model:effort，例如 auto/gpt-6-luna:low",
    )
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--out", default="runs/model.jsonl")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="", help="low/medium/high，留空则不下发")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--demo", action="store_true", help="生成 tidy/slop 两份合成样本，不调用模型")
    parser.add_argument("--out-prefix", default="runs/demo")
    args = parser.parse_args()

    if args.demo:
        tasks = load_tasks(args.tasks)
        logger.info(f"载入 {len(tasks)} 道题")
        run_demo(tasks, args.out_prefix)
        return

    specs = args.models or [args.model]
    if not args.models and args.out != "runs/model.jsonl":
        tasks = load_tasks(args.tasks)
        logger.info(f"载入 {len(tasks)} 道题")
        model, _, effort = args.model.partition(":")
        eff = effort or args.reasoning_effort
        logger.info(f"开始 {model}，思考档位 {eff or '默认'}")
        asyncio.run(
            execute_live(
                tasks,
                model=model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                out=args.out,
                concurrency=args.concurrency,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=eff,
                timeout=args.timeout,
                limit=args.limit,
            )
        )
        return

    execute_models(
        args.tasks,
        models=specs,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        out_dir=args.out_dir,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
