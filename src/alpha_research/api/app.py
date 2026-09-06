"""FastAPI interface and self-contained local research dashboard."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from alpha_research.api.jobs import JobManager, credential_available
from alpha_research.api.records import read_json
from alpha_research.api.settings import Settings


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["mock", "online"]
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    target_rounds: int = Field(default=100, ge=5, le=10000, multiple_of=5)
    interval_seconds: float = Field(default=2.5, ge=.2, le=60, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_replay(self):
        if self.mode == "mock" and (self.seed not in (42, 123, 456) or self.target_rounds > 100):
            raise ValueError("Mock supports measured seeds 42, 123, 456 through round 100")
        return self


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_rounds: int | None = Field(default=None, ge=5, le=10000, multiple_of=5)
    interval_seconds: float | None = Field(default=None, ge=.2, le=60, allow_inf_nan=False)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        settings.jobs.mkdir(parents=True, exist_ok=True)
        with (settings.jobs.parent / "service.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another AlphaLDM API service uses this runtime directory. Use one Uvicorn worker.") from exc
            app.state.manager = JobManager(settings)
            yield
            # Detached jobs continue safely; restarting the API reconnects to them.

    app = FastAPI(title="AlphaLDM Discovery API", version="1.0.0", lifespan=lifespan,
        description="Measured historical replay and live AlphaLDM search with five-round Validation Top-5 / Test checkpoints.")
    allowed = os.environ.get("ALPHALDM_API_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed)

    @app.middleware("http")
    async def protect_local_actions(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin control requests are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({"detail": "Unknown run"}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic normally echoes rejected input, which could include an
        # accidentally supplied credential. Return only field/error metadata.
        return JSONResponse({"detail": [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                                        for error in exc.errors()]}, status_code=422)

    @app.exception_handler(ValueError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "1.0.0", "checkpoint_step": 5}

    @app.get("/api/readiness")
    def readiness():
        provider = Path(os.environ.get("ALPHARESEARCH_QLIB_PROVIDER_URI",
                        str(settings.repo_root.parent / "quantaalpha_qlib_csi300/cn_data")))
        ffo_url = os.environ.get("ALPHARESEARCH_FFO_URL", "http://127.0.0.1:19777")
        try:
            with httpx.Client(timeout=3, trust_env=False) as client:
                evaluator = client.get(ffo_url.rstrip("/") + "/health").is_success
        except httpx.HTTPError:
            evaluator = False
        credential = credential_available()
        data = (provider / "calendars/day.txt").is_file()
        replays = [s for s in (42, 123, 456) if (settings.replays / f"seed{s}.json").is_file()]
        return {"online_ready": credential and data and evaluator, "credential_available": credential,
                "market_data_available": data, "evaluator_ready": evaluator, "replay_seeds": replays,
                "model": "DeepSeek V4 Flash", "checkpoint_step": 5}

    @app.get("/api/archives")
    def archives():
        items = []
        for seed in (42, 123, 456):
            archive = read_json(settings.replays / f"seed{seed}.json")
            if archive:
                items.append({"seed": seed, "rounds": 100, "checkpoint_step": 5,
                              "frames": len(archive["frames"]), "initial": archive["frames"][0]})
        return {"archives": items}

    @app.post("/api/runs", status_code=202)
    def create(body: RunRequest, request: Request):
        if body.mode == "online":
            ready = readiness()
            if not ready["online_ready"]:
                missing = [k for k in ("credential_available", "market_data_available", "evaluator_ready") if not ready[k]]
                raise HTTPException(status_code=503, detail={"message": "Online prerequisites are unavailable", "missing": missing})
        return request.app.state.manager.create(**body.model_dump())

    @app.get("/api/runs")
    def runs(request: Request):
        return {"runs": request.app.state.manager.list()}

    @app.get("/api/runs/{job_id}")
    def get_run(job_id: str, request: Request):
        return request.app.state.manager.snapshot(job_id)

    @app.post("/api/runs/{job_id}/pause", status_code=202)
    def pause(job_id: str, request: Request):
        return request.app.state.manager.pause(job_id)

    @app.post("/api/runs/{job_id}/stop")
    def stop(job_id: str, request: Request):
        """Immediately cancel this worker; keep its last durable checkpoint."""
        return request.app.state.manager.stop(job_id)

    @app.post("/api/runs/{job_id}/resume", status_code=202)
    def resume(job_id: str, body: ResumeRequest, request: Request):
        return request.app.state.manager.resume(job_id, **body.model_dump())

    @app.get("/api/runs/{job_id}/export")
    def export(job_id: str, request: Request):
        content = request.app.state.manager.snapshot(job_id)
        content["factors"] = request.app.state.manager.factors(job_id)
        return JSONResponse(content, headers={"Content-Disposition": f'attachment; filename="alphaldm-{job_id}.json"'})

    @app.get("/api/runs/{job_id}/factors")
    def factors(job_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        rows = request.app.state.manager.factors(job_id)
        return {"total": len(rows), "offset": offset, "factors": rows[offset:offset+limit]}

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Start the local AlphaLDM research dashboard")
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    uvicorn.run("alpha_research.api.app:create_app", factory=True, host="127.0.0.1", port=options.port, workers=1)


if __name__ == "__main__":
    main()
