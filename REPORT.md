# REPORT

## 1. Serving configuration (Phase 1)

Model: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 on 1x H100 80GB, vLLM 0.10.2.

| Flag | Value | Why (for this workload) |
|------|-------|--------------------------|
| model (FP8 checkpoint) | Qwen3-30B-A3B-Instruct-2507-FP8 | BF16 weights load at 56.9 GiB, leaving only 8.68 GiB for KV — too little for concurrency, and the engine refused to start. FP8 loads at 29.1 GiB, freeing 42.93 GiB for KV (a ~5x increase) and running the bandwidth-bound MoE decode natively on H100 FP8. |
| --max-model-len | 8192 | Prompts are ~1.5-3K tokens with short SQL outputs. The model default (262144) reserved 24 GiB of KV per the formula and crashed; 8192 cuts that to a fraction and lets the scheduler admit far more concurrent sequences. |
| --max-num-seqs | 128 | The SLO (10 RPS x 2-3 dependent calls x ~1-2s each) implies ~20-60 concurrent sequences; measured max concurrency at 8192 tokens is 57.2x. 128 gives headroom for bursts without inflating inter-token latency. |
| --gpu-memory-utilization | 0.92 | Claims most of the 80 GB card for KV while leaving room for activations and CUDA graphs (peak activation 0.75 GiB, CUDA graph 0.45 GiB observed). |
| --enable-prefix-caching | on | The rendered DB schema is an identical prompt prefix across a request's 2-3 agent calls and across requests hitting the same DB; caching it cuts prefill/TTFT. |

Measured at startup (FP8): weights 29.1 GiB, KV cache 42.93 GiB available,
GPU KV cache size 468,864 tokens, max concurrency 57.2x at 8192 tokens/request.
Manual query verified: prompt "how many singers are there?" returned
`SELECT COUNT(*) FROM singer;` (see screenshots/vllm_manual_query.jpg).

## 2. Monitoring dashboard (Phase 2)

The Grafana dashboard groups vLLM metrics into three diagnostic rows -
Latency (E2E, TTFT, queue + per-output-token), Throughput (running/waiting,
finished/s, token throughput), and KV cache (usage, preemptions, prefix-cache
hit rate) - so a single view answers "is it slow, and where in the lifecycle?":
queue time rising first points to scheduler saturation, per-output-token
rising points to an oversized decode batch. Under a manual burst the prefix-
cache hit rate reached ~82.5%, confirming the schema/prompt prefix is being
reused across requests as the Phase 1 prefix-caching flag intended.
See screenshots/grafana_serving.jpg.

## 3. Agent (Phase 3)

LangGraph text-to-SQL agent with a generate -> execute -> verify -> revise loop,
capped at MAX_ITERATIONS=3 (one generate + up to two revises). The verifier asks
the model for a {"ok", "issue"} verdict and fails open on unparseable replies, so
a flaky verifier degrades to "no verifier" rather than burning revise iterations.
Prompts enforce SQLite dialect and column discipline (select only what the
question asks) since the Phase 5 eval compares executed row sets.

## 4. Tracing (Phase 4)

Langfuse captures every agent run as a LangGraph trace with the full nested-span waterfall 
- `attach_schema -> generate_sql -> execute -> verify -> revise -> execute -> verify` - each 
span carrying its prompt, model response, latency, and token counts (e.g. the verifier span: 351->23 tokens, 0.17 s). 
Tracing is wired through the langfuse v4 `CallbackHandler` (`from langfuse.langchain import CallbackHandler` in `agent/server.py`); 
the README's `langfuse.callback` snippet is the v2 API and would not import against the pinned langfuse 4.7.1, so it was not used. 
A trace of the "average number of crimes committed in 1995..." question (db `financial`) demonstrates the verify->revise loop earning its 
place: the first generated SQL returned 0 rows, the verifier rejected it (`ok=false`, "0 rows were returned but the question clearly implies 
matching rows exist"), and the agent revised twice before terminating - captured in `screenshots/langfuse_trace.png`. 
Per-request `tags` are forwarded by the server as a Langfuse `langfuse_tags` list (not merely run metadata), 
so the trace list is filterable by run label rather than requiring a free-text metadata search. 
`screenshots/langfuse_tags.png` shows the list filtered to `run:phase4-demo`. 
Both runners expose a `--tag` flag (`run_eval.py --tag eval-baseline`, `driver.py --tag load-iter2`) so Phase 6 load traces can be isolated 
per iteration and the slow requests from a specific run inspected directly.


## 5. Evals (Phase 5)

Run: 30-question BIRD-style set, agent serving Qwen3-30B via vLLM, full
Langfuse trace capture. Wall clock 29.1s. Output: results/eval_baseline.json.

Headline numbers:
  - Execution accuracy (final):  0.3667  (11/30)
  - Pass rate by iteration:      [0.3667, 0.3667, 0.3667]
  - Questions triggering revise: 9 / 30
  - Mean attempts per question:  1.533  (across all 30)
  - Agent errors / gold failures: 0 / 0

Finding: the revise loop runs but never converts a wrong answer to a right
one - and most of the time it does nothing at all. The per-iteration curve
is flat because revising 9 questions changed zero outcomes. The attempt log
splits the 30 questions into four buckets:

  - Correct on first pass ............................ 11
  - Wrong, never revised (accepted silently) ........ 10
  - Wrong, revised -> identical SQL re-emitted ....... 6
  - Wrong, revised -> changed but still wrong ........ 3

Two diagnostic points follow:

1. The verifier is not error-driven. Every query executes cleanly
   (exec_ok = true everywhere), so revise is never triggered by an
   execution error - only by a result-level check. That check catches
   just 9 of the 19 wrong answers; the other 10 executable-but-wrong
   queries pass through unflagged. (Exact trigger condition to be
   confirmed against the verifier in agent/graph.py.)

2. When revise does fire, it has no correctness signal to act on. There
   is no gold comparison and no error message fed back, so the prompt can
   only say "try again." In 6 of 9 cases the model re-emitted the
   identical query (e.g. the Art-and-Design-students question regenerated
   the same SQL three times). In the 3 cases where it changed the SQL,
   the edits were plausible but blind: the Hamilton lap-time query
   switched time parsing from H:M:S to M:S; the mythic-gladiator query
   changed format = 'gladiator' to 'Gladiator' - a correct guess that
   casing might be the bug, but still wrong. The loop can reason; it
   cannot converge without feedback.

Where the accuracy ceiling sits: the misses are generation-time grounding
errors, not loop failures - value/encoding mismatches ('Cl' vs 'Chlorine',
format casing), guessed "normal" thresholds (IgG <= 100, UA <= 7.0),
join-cardinality inflation, and semantic misreadings (counting not-finished
races as disqualified). A result-aware verifier (gold-output comparison or
self-consistency / schema grounding) would make the revise signal correlate
with correctness rather than non-emptiness - the single change most likely
to move the curve. (Out of scope for this phase.)

Monitoring during the eval: screenshots/grafana_eval_run.png shows the vLLM
serving dashboard tracking the pipeline end-to-end while the eval is in
flight (Last 5 min, 5s refresh). The 29s run appears as the burst at
~15:30:45-15:32:00, and Requests running peaks at 1 - confirming serial
execution, consistent with ~1 q/s for 30 questions in 29s.

  - Latency: E2E p50 ~300ms, p95/p99 ~470-500ms; TTFT p50/p95 ~15-19ms;
    queue p95 ~270ms. End-to-end time is dominated by queueing, not
    first-token compute - expected for short serial requests.
  - Throughput: finished rate rises to several req/s; prompt-token
    throughput peaks ~3K tok/s against low generation-token throughput,
    consistent with text-to-SQL (long schema prompt, short completion).
  - KV cache: usage ~0%, preemptions 0. At concurrency 1 the cache is
    barely touched - healthy headroom, no eviction.
  - Prefix cache hit rate ~100% once traffic starts - the shared schema /
    system prefix across all 30 questions is being reused as intended.

Artifacts: results/eval_baseline.json, screenshots/grafana_eval_run.png,
Langfuse traces tagged run:phase4-demo.
