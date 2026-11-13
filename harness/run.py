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

THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)


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
) -> Sample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            text, think_chars = strip_think(content)
            return Sample(
                task_id=task.id,
                model=model,
                response=text,
                think_chars=think_chars,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        except Exception as exc:  # noqa: BLE001 - 网络重试需要兜底
            last_error = exc
            await asyncio.sleep(2**attempt)
    logger.error(f"{task.id} 调用失败：{last_error}")
    return Sample(task_id=task.id, model=model, response="")


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
            },
            ensure_ascii=False,
        )
        for s in samples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.success(f"写出 {len(samples)} 条 -> {path}")


def run_demo(tasks: dict[str, Task], prefix: str) -> None:
    tidy = [Sample(t, "demo-tidy", tidy_response(task)) for t, task in tasks.items()]
    slop = [Sample(t, "demo-slop", slop_response(task)) for t, task in tasks.items()]
    base = Path(prefix)
    write_jsonl(base.with_name(base.name + "_tidy.jsonl"), tidy)
    write_jsonl(base.with_name(base.name + "_slop.jsonl"), slop)


async def run_live(args: argparse.Namespace, tasks: dict[str, Task]) -> None:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"环境变量 {args.api_key_env} 为空")
    semaphore = asyncio.Semaphore(args.concurrency)
    selected = list(tasks.values())[: args.limit] if args.limit else list(tasks.values())

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def worker(task: Task) -> Sample:
            async with semaphore:
                return await call_once(
                    client, args.base_url, api_key, args.model, task, args.temperature, args.max_tokens
                )

        samples = await asyncio.gather(*(worker(t) for t in selected))
    write_jsonl(Path(args.out), list(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="TshuBench 调用器")
    parser.add_argument("--tasks", default="tasks/eval_suite_v2.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--out", default="runs/model.jsonl")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--demo", action="store_true", help="生成 tidy/slop 两份合成样本，不调用模型")
    parser.add_argument("--out-prefix", default="runs/demo")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    logger.info(f"载入 {len(tasks)} 道题")

    if args.demo:
        run_demo(tasks, args.out_prefix)
        return

    asyncio.run(run_live(args, tasks))


if __name__ == "__main__":
    main()
