#!/usr/bin/env bash
#
# Start vLLM tuned for this workload (1x H100 80GB, Qwen3-30B-A3B MoE).
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Single source of truth: .env (same file the agent reads).
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --disable-log-requests
