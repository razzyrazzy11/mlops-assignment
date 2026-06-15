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
