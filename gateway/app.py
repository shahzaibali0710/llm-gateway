"""FastAPI application: OpenAI-compatible chat completions with routing,
fallback, guardrails, caching and metrics.

Run with:  uvicorn --factory gateway.app:create_app --port 8080
"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from .cache import ResponseCache, make_key
from .config import Config, load_config
from .guardrails import GuardrailError, Guardrails, redact_messages
from .metrics import Metrics
from .providers import Completion, Provider, ProviderError, build_providers
from .routing import UnknownRouteError, is_known, resolve

logger = logging.getLogger("gateway")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    cache: bool = True
    stream: bool = False


class GatewayError(Exception):
    def __init__(self, status: int, message: str, kind: str, headers: dict[str, str] | None = None) -> None:
        self.status, self.message, self.kind, self.headers = status, message, kind, headers or {}


def _setup_logging() -> None:
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def create_app(config: Config | None = None, providers: dict[str, Provider] | None = None) -> FastAPI:
    cfg = config or load_config()
    _setup_logging()
    metrics = Metrics()
    guardrails = Guardrails(cfg.guardrails)
    cache = ResponseCache(cfg.cache.ttl_seconds, cfg.cache.max_entries, cfg.cache.enabled)
    provs = providers or build_providers(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for p in provs.values():
            await p.aclose()

    app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)
    app.state.config, app.state.metrics, app.state.cache = cfg, metrics, cache

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": exc.message, "type": exc.kind}}, status_code=exc.status, headers=exc.headers
        )

    def authenticate(authorization: str | None = Header(default=None)) -> str:
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip().encode()
            team = None
            for key, name in cfg.api_keys.items():  # constant-time compare, no early exit
                if hmac.compare_digest(presented, key.encode()):
                    team = name
            if team:
                return team
        raise GatewayError(
            401, "missing or invalid API key", "authentication_error", {"WWW-Authenticate": "Bearer"}
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/usage")
    async def usage(_: str = Depends(authenticate)) -> dict:
        return {"teams": metrics.usage_snapshot()}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, team: str = Depends(authenticate)) -> dict:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        messages = [m.model_dump() for m in req.messages]
        # Never use client-controlled strings as metric labels.
        label = req.model if is_known(req.model, cfg) else "invalid"

        if req.stream:
            raise GatewayError(400, "streaming is not supported yet", "invalid_request_error")

        try:
            guardrails.check(messages)
        except GuardrailError as exc:
            metrics.record_request(team, label, "none", "blocked")
            logger.warning("blocked id=%s team=%s reason=%s", request_id, team, exc)
            raise GatewayError(400, str(exc), "guardrail_error") from exc

        try:
            resolution = resolve(req.model, messages, cfg)
        except UnknownRouteError:
            metrics.record_request(team, label, "none", "bad_request")
            known = ["auto", *cfg.routes, *cfg.models]
            raise GatewayError(
                400, f"unknown model/route {req.model!r}; use one of {known}", "invalid_request_error"
            ) from None

        route, reason = resolution.route, resolution.reason
        use_cache = req.cache and cache.enabled
        key = make_key(route, messages, req.max_tokens)

        completion: Completion | None = None
        used_model = ""
        cost = 0.0
        failed: list[str] = []
        cached = False

        hit = cache.get(key) if use_cache else None
        if hit is not None:
            completion, used_model = hit
            cached = True
            metrics.record_cache_hit(team)
        else:
            chain = resolution.chain
            for i, name in enumerate(chain):
                model_cfg = cfg.models[name]
                provider = provs[model_cfg.provider]
                outgoing = messages
                if provider.external:
                    outgoing, found = redact_messages(messages)
                    for kind, n in found.items():
                        metrics.pii_redactions.labels(kind).inc(n)
                t0 = time.perf_counter()
                try:
                    completion = await provider.complete(model_cfg.model, outgoing, req.max_tokens)
                except ProviderError as exc:
                    metrics.latency.labels(name).observe(time.perf_counter() - t0)
                    failed.append(name)
                    logger.warning("provider failed id=%s model=%s error=%s", request_id, name, exc)
                    if i + 1 < len(chain):
                        metrics.fallbacks.labels(route, name).inc()
                    continue
                metrics.latency.labels(name).observe(time.perf_counter() - t0)
                used_model = name
                cost = model_cfg.cost(completion.input_tokens, completion.output_tokens)
                break

            if completion is None:
                metrics.record_request(team, route, "none", "error")
                logger.error(
                    "failed id=%s team=%s route=%s reason=%r tried=%s", request_id, team, route, reason, failed
                )
                raise GatewayError(
                    502, f"all models in route {route!r} failed: {failed}", "upstream_error"
                )
            metrics.record_success(team, used_model, completion.input_tokens, completion.output_tokens, cost)
            if use_cache:
                cache.put(key, (completion, used_model))

        metrics.record_request(team, route, used_model, "ok")
        latency = time.perf_counter() - started
        logger.info(
            'request id=%s team=%s route=%s reason="%s" model=%s latency_s=%.3f cost_usd=%.6f cached=%s fallbacks=%d',
            request_id, team, route, reason, used_model, latency, cost, cached, len(failed),
        )
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": completion.input_tokens,
                "completion_tokens": completion.output_tokens,
                "total_tokens": completion.input_tokens + completion.output_tokens,
            },
            "gateway": {
                "request_id": request_id,
                "route": route,
                "reason": reason,
                "latency_s": round(latency, 4),
                "cost_usd": round(cost, 8),
                "fallbacks": failed,
                "cached": cached,
            },
        }

    return app
