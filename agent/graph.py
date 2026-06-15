from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3 = one generate + up to two revises. Raising it buys little quality
# (per-iteration pass rate flattens) and costs P95 latency; tune with data.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=4)
def llm(max_tokens: int = 512) -> ChatOpenAI:
    """Cached chat client pointed at VLLM_BASE_URL.

    Cached (one client per max_tokens value) so every request reuses the
    same pooled HTTP connections instead of opening a new client per node
    call. timeout/max_retries are explicit so a wedged backend fails fast
    instead of silently retrying into the latency budget.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=1,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```",
                       text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verify_reply(text: str) -> tuple[bool, str]:
    """Defensively parse {"ok": bool, "issue": str} out of an LLM reply.

    Fail OPEN (ok=True) on unparseable output: a broken verifier should not
    burn extra revise iterations (and latency) on every request. The parse
    failure is recorded in `issue` so it shows up in Langfuse traces.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return True, "verify_parse_failed: no JSON object in reply"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return True, "verify_parse_failed: invalid JSON"
    return bool(obj.get("ok", True)), str(obj.get("issue", "")).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Async so the FastAPI server can run many agent graphs concurrently on
    one event loop (see module docstring, change #2).
    """
    response = await llm().ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result.

    Left sync on purpose: sqlite on local disk over small BIRD DBs is
    fast, and LangGraph runs sync nodes in a worker thread under ainvoke.
    """
    return {"execution": execute_sql(state.db_id, state.sql)}


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Asks the model for {"ok": bool, "issue": str}; parsed defensively by
    _parse_verify_reply. The verifier sees the question, the SQL, and the
    compact execution rendering (rows / 0 rows / ERROR), so it can catch
    the three target failure cases from the README: SQL errored, zero rows
    when rows are implied, and columns that don't answer the question.

    max_tokens=160: the reply is a one-line JSON object; capping it keeps
    the per-request decode budget (and P95) down.
    """
    rendered = state.execution.render() if state.execution else "NO EXECUTION RESULT"
    response = await llm(max_tokens=160).ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=rendered,
        )),
    ])
    ok, issue = _parse_verify_reply(response.content)
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node; the prompt additionally carries the
    failing SQL, its execution result, and the verifier's complaint.
    Bumps `iteration` so route_after_verify can enforce MAX_ITERATIONS,
    and appends to history so the eval runner can score every attempt
    (per-iteration pass rate, Phase 5).
    """
    rendered = state.execution.render() if state.execution else "NO EXECUTION RESULT"
    response = await llm().ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=rendered,
            issue=state.verify_issue or "(verifier gave no detail)",
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    End when the verifier is satisfied OR the generate/revise budget is
    spent. iteration counts LLM *SQL-producing* calls (generate=1, each
    revise +1), so MAX_ITERATIONS=3 means at most 2 revise passes.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
