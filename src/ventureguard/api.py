"""Public demo API. No account, uploaded files, LLM or stored user profiles."""
from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from .data import ROOT
from .service import ResearchService
from .simulation import SimulationRequest, simulate


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["core13", "full64"] = "core13"
    example_id: str | None = Field(default=None, max_length=40)
    features: dict[str, FiniteFloat | None] | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def one_input(self):
        if (self.example_id is None) == (self.features is None):
            raise ValueError("Supply exactly one of example_id or features.")
        return self


@asynccontextmanager
async def lifespan(app):
    app.state.research = ResearchService()
    yield


app = FastAPI(title="VentureGuard", version="1.0.0", lifespan=lifespan,
              description="Financial ML research and cash-flow planning. RAG deferred.")
origins = [v.strip() for v in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if v.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
requests_seen = deque()


@app.middleware("http")
async def demo_limits(request: Request, call_next):
    if request.url.path.startswith("/v1") and request.method == "POST":
        now = time.monotonic()
        while requests_seen and requests_seen[0] < now - 60:
            requests_seen.popleft()
        if len(requests_seen) >= 120:
            return JSONResponse(status_code=429, content={"detail": "Demo is busy. Try again in a minute."}, headers={"Retry-After": "60"})
        requests_seen.append(now)
        # Bound actual streamed bytes as well as Content-Length.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse(status_code=413, content={"detail": "Request exceeds the 64 KB demo limit."})
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "research"), "version": "1.0.0"}


@app.get("/v1/model-info")
def model_info():
    return app.state.research.metadata


@app.get("/v1/examples")
def examples():
    return app.state.research.example_list()


@app.post("/v1/benchmark/predict")
def predict(request: PredictionRequest):
    try:
        return app.state.research.predict(request.schema_name, request.features, request.example_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/simulate")
def simulation(request: SimulationRequest):
    return simulate(request)


# A same-origin build can be served by the same container; Static Web Apps is
# also supported by setting VITE_API_URL at frontend build time.
dist = ROOT / "frontend/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
