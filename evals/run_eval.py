"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from agent.graph import MAX_ITERATIONS

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    
    # Gold first: run it so a broken gold query is recorded, not fatal
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # Call the agent.
    record = {
       "db_id": db_id,
       "question": question["question"],
       "gold_exec_ok": gold_ok,
       "gold_error": gold_err,
       "agent_error": None,
       "attempts": [],        # one per SQL-producing node, in order
       "iter_correct": [],    # bool per iteration index (0-based)
       "terminated_at": None, # index of last SQL actually emitted
}

    try:
        resp = httpx.post(
	agent_url, 
	json={"question": question["question"], "db": db_id}, 
	timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

    except Exception as e:
        record["agent_error"] = f"{type(e).__name__}: {e}"
        return record

    history = data.get("history", []) or []

    # Extract SQL-producing steps in order: generate_sql (idx 0) then each revise.
    sql_steps = [h for h in history if h.get("node") in ("generate_sql", "revise")]

    for idx, step in enumerate(sql_steps):
        sql = step.get("sql") or ""
        ok, rows, err = run_sql(db_id, sql)
        correct = matches(gold_rows, rows) if (gold_ok and ok) else False
        record["attempts"].append({
            "iteration": idx,
            "node": step.get("node"),
            "sql": sql,
            "exec_ok": ok,
            "exec_error": err,
            "correct": correct,
        })
        record["iter_correct"].append(correct)

    record["terminated_at"] = len(sql_steps) - 1 if sql_steps else None
    return record

def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    max_iters = MAX_ITERATIONS

    # Per-iteration pass rate with carry-forward: a question that terminated
    # at iteration j contributes its iteration-j result to all k >= j.
    iter_pass_counts = [0] * max_iters
    for r in results:
        ic = r["iter_correct"]
        if not ic:
            continue  # agent error / no SQL emitted -> contributes 0 everywhere
        for k in range(max_iters):
            # carry forward the last emitted iteration if k beyond termination
            val = ic[k] if k < len(ic) else ic[-1]
            if val:
                iter_pass_counts[k] += 1

    pass_rate_by_iteration = [round(c / n, 4) for c in iter_pass_counts] if n else []

    final_correct = sum(1 for r in results if r["iter_correct"] and r["iter_correct"][-1])
    questions_with_revise = sum(1 for r in results if len(r["iter_correct"]) > 1)
    agent_errors = sum(1 for r in results if r["agent_error"])
    gold_sql_failures = sum(1 for r in results if not r["gold_exec_ok"])
    mean_iterations = round(
        sum(len(r["iter_correct"]) for r in results) / n, 3) if n else 0.0

    return {
        "n_questions": n,
        "execution_accuracy_final": round(final_correct / n, 4) if n else 0.0,
        "pass_rate_by_iteration": pass_rate_by_iteration,
        "iteration_indexing": "index 0 = initial generate_sql; index k = after k revise passes; carry-forward applied",
        "questions_with_revise": questions_with_revise,
        "agent_errors": agent_errors,
        "gold_sql_failures": gold_sql_failures,
        "mean_iterations": mean_iterations,
    }

# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
