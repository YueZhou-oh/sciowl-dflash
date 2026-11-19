#!/usr/bin/env python3
"""Run one task: N concurrent generations, print client throughput as JSON."""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks_6k import TASKS


def one(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300], "dt": time.perf_counter() - t0}
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "dt": time.perf_counter() - t0,
        "prompt": int(usage.get("prompt_tokens") or 0),
        "completion": int(usage.get("completion_tokens") or 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    prompt = dict(TASKS)[args.task]
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = args.url.rstrip("/") + "/v1/chat/completions"
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(one, url, payload, args.timeout) for _ in range(args.requests)
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    prompt_toks = sum(r["prompt"] for r in ok)
    gen_toks = sum(r["completion"] for r in ok)
    row = {
        "task": args.task,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "ok": len(ok),
        "errors": len(bad),
        "error_sample": bad[0]["error"] if bad else "",
        "wall_s": round(wall, 2),
        "prompt_tokens": prompt_toks,
        "completion_tokens": gen_toks,
        "mean_completion": round(gen_toks / len(ok), 1) if ok else 0,
        "prompt_tps": round(prompt_toks / wall, 1) if wall else 0,
        "generation_tps": round(gen_toks / wall, 1) if wall else 0,
    }
    Path(args.out).write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
