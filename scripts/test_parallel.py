#!/usr/bin/env python3
"""Run Mol-Instructions through an OpenAI-compatible API (Python >= 3.9).

Install: pip install openai httpx
Run: python test_parallel.py --input test.jsonl --output predictions.jsonl

Sampling settings match the supplied working test.py, with 48 concurrent requests.
Output limits: MG/MC/FS/RP/RS = 256 tokens; PP = 32 tokens.
The optional --tokenizer /path/to/server/tokenizer requires transformers and
renders the saved `prompt` with that tokenizer. Without it, `prompt` is a
reconstruction of the Qwen-style template in the supplied predictions file.
The API always receives messages, never the rendered prompt.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError


TASKS = {
    "description_guided_molecule_design": "MG",
    "forward_reaction_prediction": "FS",
    "molecular_description_generation": "MC",
    "property_prediction": "PP",
    "reagent_prediction": "RP",
    "retrosynthesis": "RS",
}


def load_samples(path):
    samples = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for key in ("instruction", "input", "output", "source_file", "source_index"):
                    if key not in row:
                        raise ValueError(f"missing {key!r}")
                for key in ("instruction", "input", "source_file"):
                    if not isinstance(row[key], str):
                        raise ValueError(f"{key} must be a string")
                if Path(row["source_file"]).stem not in TASKS:
                    raise ValueError(f"unknown task source: {row['source_file']}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            samples.append(row)
    return samples


def make_record(index, row, max_tokens, tokenizer=None):
    # Preserve the strings exactly, including whitespace; join with one newline.
    content = row["instruction"] + "\n" + row["input"] + "\ngive me the final answer directly."
    messages = [{"role": "user", "content": content}]
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    else:
        prompt = (
            "user\n" + content + "\n"
            "assistant\n"
        )
    return messages, {
        "id": index,
        "task": TASKS[Path(row["source_file"]).stem],
        "instruction": row["instruction"],
        "input": row["input"],
        "reference": row["output"],
        "source_file": row["source_file"],
        "source_index": row["source_index"],
        "prediction": "",
        "status": "error",
        "prompt_tokens": None,
        "max_new_tokens": max_tokens,
        "prompt": prompt,
    }


async def predict(client, model, messages, record, args):
    for attempt in range(args.retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=record["max_new_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if not response.choices or response.choices[0].message.content is None:
                raise ValueError("API response has no text content")
            record["prediction"] = response.choices[0].message.content
            record["status"] = "ok"
            if response.usage is not None:
                record["prompt_tokens"] = response.usage.prompt_tokens
            # A response stopped at the token limit is still a valid prediction,
            # consistent with the supplied predictions.jsonl.
            return record
        except (APIConnectionError, APIStatusError, ValueError) as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = isinstance(exc, APIConnectionError) or (
                status_code in (408, 409, 429)
                or (status_code is not None and status_code >= 500)
            )
            if retryable and attempt < args.retries:
                delay = min(60.0, 2.0 ** attempt) + random.random()
                print(
                    f"id={record['id']}: retry {attempt + 1}/{args.retries} "
                    f"in {delay:.1f}s ({type(exc).__name__}, HTTP {status_code})",
                    file=sys.stderr, flush=True,
                )
                await asyncio.sleep(delay)
                continue
            print(f"id={record['id']}: FAILED: {exc}", file=sys.stderr, flush=True)
            return record


async def run(args):
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"{args.output} already exists; choose another path or use --overwrite")
    samples = load_samples(args.input)
    if args.limit is not None:
        samples = samples[:args.limit]
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    # Prepare/validate prompts before creating the output file.
    prepared = [
        make_record(
            i, row,
            getattr(args, "mol_max_tokens_" + TASKS[Path(row["source_file"]).stem].lower()),
            tokenizer,
        )
        for i, row in enumerate(samples)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    errors = 0
    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=args.concurrency,
        ),
        timeout=args.timeout,
    ) as http_client:
        async with AsyncOpenAI(
            api_key=args.api_key, base_url=args.base_url,
            http_client=http_client, max_retries=0, timeout=args.timeout,
        ) as client:
            model = args.model
            if not model:
                models = await client.models.list()
                if not models.data:
                    raise ValueError("API returned no models; specify --model")
                model = models.data[0].id
            print(f"Model={model}; samples={len(samples)}; concurrency={args.concurrency}",
                  file=sys.stderr, flush=True)
            jobs = asyncio.Queue()
            results = asyncio.Queue()
            for index, (messages, record) in enumerate(prepared):
                jobs.put_nowait((index, messages, record))

            async def worker():
                while True:
                    try:
                        index, messages, record = jobs.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        result = await predict(client, model, messages, record, args)
                        await results.put((index, result, None))
                    except Exception as exc:
                        # Surface unexpected programming/server response errors to
                        # the writer, so it cannot hang waiting for this index.
                        await results.put((index, None, exc))

            workers = [asyncio.create_task(worker())
                       for _ in range(min(args.concurrency, len(samples)))]
            pending = {}
            next_id = 0
            try:
                with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as f:
                    for completed in range(1, len(samples) + 1):
                        index, record, exc = await results.get()
                        if exc is not None:
                            raise RuntimeError(f"Unexpected error for id={index}: {exc}") from exc
                        errors += record["status"] != "ok"
                        pending[index] = record
                        # Responses finish out of order; write original input order.
                        while next_id in pending:
                            f.write(json.dumps(pending.pop(next_id), ensure_ascii=False) + "\n")
                            next_id += 1
                        f.flush()
                        if completed % 50 == 0 or completed == len(samples):
                            elapsed = time.monotonic() - started
                            print(f"Completed {completed}/{len(samples)}; errors={errors}; "
                                  f"{completed / max(elapsed, 0.001):.2f} samples/s",
                                  file=sys.stderr, flush=True)
            finally:
                for task in workers:
                    task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
    print(f"Saved {len(samples)} rows to {args.output}; failed={errors}", file=sys.stderr)
    return 1 if errors else 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("./test.jsonl"))
    p.add_argument("--output", type=Path, default=Path("./speed_baseline_predictions.jsonl"))
    p.add_argument("--base-url", default="http://10.1.0.232:8005/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", help="Default: first model returned by /v1/models")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--mol-max-tokens-mg", type=int, default=256)
    p.add_argument("--mol-max-tokens-mc", type=int, default=256)
    p.add_argument("--mol-max-tokens-fs", type=int, default=256)
    p.add_argument("--mol-max-tokens-rp", type=int, default=256)
    p.add_argument("--mol-max-tokens-rs", type=int, default=256)
    p.add_argument("--mol-max-tokens-pp", type=int, default=32)
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="Extra attempts for transient failures")
    p.add_argument("--limit", type=int, help="Run only the first N samples")
    p.add_argument("--tokenizer", help="Optional local server tokenizer directory for exact saved prompt")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not 1 <= args.concurrency <= 64:
        p.error("--concurrency must be between 1 and 64")
    for task in TASKS.values():
        option = "mol_max_tokens_" + task.lower()
        if getattr(args, option) < 1:
            p.error("--" + option.replace("_", "-") + " must be positive")
    if args.timeout <= 0 or args.retries < 0:
        p.error("timeout must be positive; retries must be nonnegative")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    return args


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        print("Interrupted; output contains the written input-order prefix.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
