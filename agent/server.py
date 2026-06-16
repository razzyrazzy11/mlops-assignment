from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler  # v3/v4 import path

    _lf_handler = CallbackHandler()


app = FastAPI()
@app.exception_handler(Exception)
async def all_exceptions(request: Request, exc: Exception):
    import traceback; traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    # langfuse_tags -> filterable tags in the Langfuse UI; the raw dict is
    # also kept so individual key/value pairs land in trace metadata.
    metadata: dict[str, Any] = dict(req.tags)
    if req.tags:
        metadata["langfuse_tags"] = [f"{k}:{v}" for k, v in req.tags.items()]
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": metadata,
    }
    try:
        final = await graph.ainvoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
