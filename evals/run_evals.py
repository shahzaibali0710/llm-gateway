"""Run the newsroom eval cases against the gateway, per target.

    python evals/run_evals.py --gateway http://localhost:8080 \\
        --targets local-small claude-fast auto

Every case is sent with the cache disabled. Scoring is rule-based:
  must_contain      every entry must appear (case-insensitive); "a|b" means either
  must_not_contain  no entry may appear;                          "a|b" means neither
  max_words         upper bound on the answer length
Targets are any model, route or "auto" the gateway knows. Output: a PASS/FAIL line
per case, a summary table, and evals/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DEFAULT_SYSTEM = (
    "You are a precise assistant for a Norwegian newsroom. Follow the instructions exactly. "
    "Be concise, never invent facts, and answer in the language the task asks for."
)


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score(case: dict, answer: str) -> list[str]:
    """Return a list of failure descriptions (empty means PASS)."""
    low = answer.lower()
    failures: list[str] = []
    for entry in case.get("must_contain", []):
        if not any(alt.lower() in low for alt in entry.split("|") if alt):
            failures.append(f"missing {entry!r}")
    for entry in case.get("must_not_contain", []):
        hit = next((alt for alt in entry.split("|") if alt and alt.lower() in low), None)
        if hit:
            failures.append(f"contains forbidden {hit!r}")
    max_words = case.get("max_words")
    if max_words is not None and len(answer.split()) > max_words:
        failures.append(f"{len(answer.split())} words > {max_words}")
    return failures


def run_case(client: httpx.Client, gateway: str, key: str, target: str, system: str, case: dict) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": case["prompt"]}]
    started = time.perf_counter()
    try:
        resp = client.post(
            f"{gateway}/v1/chat/completions",
            json={"model": target, "messages": messages, "cache": False},
            headers={"Authorization": f"Bearer {key}"},
        )
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            return {"answer": "", "latency": elapsed, "cost": 0.0, "model": "", "failures": [f"HTTP {resp.status_code}: {resp.text[:120]}"]}
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "latency": elapsed,
            "cost": data["gateway"]["cost_usd"],
            "model": data["model"],
            "failures": score(case, answer),
        }
    except httpx.HTTPError as exc:
        return {"answer": "", "latency": time.perf_counter() - started, "cost": 0.0, "model": "", "failures": [f"request error: {exc}"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--key", default="newsroom-dev-key")
    ap.add_argument("--targets", nargs="+", default=["local-small", "claude-fast", "auto"])
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results.csv")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    gateway = args.gateway.rstrip("/")
    cases = load_cases(args.cases)
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    with httpx.Client(timeout=300) as client:
        for target in args.targets:
            print(f"\n== target: {target} ==")
            results = []
            for case in cases:
                r = run_case(client, gateway, args.key, target, args.system, case)
                results.append(r)
                status = "PASS" if not r["failures"] else "FAIL"
                extra = "" if not r["failures"] else "  <- " + "; ".join(r["failures"])
                print(f"  {status}  {case['id']:<18} {r['latency']:>6.2f}s  {r['model'] or '-':<13}{extra}")
                rows.append(
                    {
                        "target": target,
                        "case": case["id"],
                        "passed": not r["failures"],
                        "model": r["model"],
                        "latency_s": round(r["latency"], 3),
                        "cost_usd": r["cost"],
                        "failures": "; ".join(r["failures"]),
                        "answer": r["answer"].replace("\n", "\\n"),
                    }
                )
            passed = sum(1 for r in results if not r["failures"])
            summary[target] = {
                "pass": passed,
                "total": len(results),
                "p50": statistics.median(r["latency"] for r in results),
                "cost": sum(r["cost"] for r in results),
            }

    print("\n" + f"{'target':<16} {'pass rate':>12} {'p50 latency':>13} {'total cost':>13}")
    print("-" * 58)
    for target, s in summary.items():
        rate = f"{s['pass']}/{s['total']} ({s['pass'] / s['total']:.0%})"
        print(f"{target:<16} {rate:>12} {s['p50']:>12.2f}s ${s['cost']:>11.6f}")

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
