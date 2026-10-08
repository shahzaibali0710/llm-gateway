# llm-gateway

A self-hostable, **OpenAI-compatible gateway** for organisations where many teams use AI. One endpoint in front of local models (Ollama / vLLM) and hosted ones (Claude): it authenticates each team, routes every request to the cheapest model that can do the job, falls back when a model is down, keeps sensitive text away from external providers, and tells you exactly who spent what.

The demo scenario is a Norwegian newsroom with three teams (`newsroom`, `product`, `data-science`) that must keep unpublished articles inside the cluster, but would like to use a strong hosted model for the hard stuff.

## Architecture

```
  newsroom / product / data-science  (any OpenAI client, Bearer API key)
                     |
                     v
        +------------------------------+
        |         llm-gateway          |
        |  auth -> team                |
        |  guardrails (size, injection)|
        |  router  (auto / route / model) --> fallback chain
        |  cache (exact match, TTL/LRU)|
        |  PII redaction (external only)|
        |  metrics + structured logs   |
        +--------+------------+--------+
                 |            |
      local, stays in cluster |   external (prompt redacted)
                 v            v
        Ollama / vLLM      Anthropic API
        qwen2.5:3b         claude-haiku-4-5
        llama3.1:8b        claude-sonnet-5-5

  Prometheus <-- /metrics        Grafana <-- Prometheus        alerts.yml
```

## Features

| Feature | What it does | File |
|---|---|---|
| OpenAI-compatible API | `POST /v1/chat/completions`, plus a `gateway` block (route, reason, latency, cost, fallbacks, cached) | `gateway/app.py` |
| Team auth | Bearer API key maps to a team; 401 otherwise | `gateway/app.py`, `config.yaml` |
| Providers | Ollama/vLLM/SGLang (OpenAI-compatible), Anthropic Messages API, offline mock | `gateway/providers.py` |
| Routes and fallback | Ordered chains per route; every model is also a single-model route; 502 only if all fail | `gateway/routing.py`, `gateway/app.py` |
| Auto-router | `model: "auto"` picks `simple` or `complex` with a transparent regex heuristic and returns the reason | `gateway/routing.py` |
| Guardrails | Max input size and blocked prompt-injection patterns (400) | `gateway/guardrails.py` |
| PII redaction | Email, fødselsnummer, card and Norwegian phone numbers, only for external providers | `gateway/guardrails.py` |
| Cache | Exact match on route + messages + max_tokens, TTL + LRU, per-request opt-out | `gateway/cache.py` |
| Metrics and usage | Prometheus counters/histograms by team and model, `GET /v1/usage` per-team totals | `gateway/metrics.py` |
| Config | One YAML file: models and prices, routes, keys, limits | `gateway/config.py`, `config.yaml` |
| Offline tests | 48 tests, mock provider only | `tests/test_gateway.py` |
| Evals | 12 newsroom cases, pass rate / p50 latency / cost per target | `evals/` |
| Deployment | Docker, compose with Prometheus + Grafana, Kubernetes manifests, GitHub Actions | `Dockerfile`, `docker-compose.yml`, `monitoring/`, `k8s/`, `.github/` |

## Quick start (Docker Compose)

Prerequisites: Docker Desktop (WSL2 backend on Windows).

```bash
# 1. (optional) enable the hosted Claude models; without a key those fall back to local models
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."

# 2. start everything
docker compose up -d --build

# 3. pull the local models into the Ollama volume (one time, a few GB)
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull llama3.1:8b

# 4. talk to it
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer newsroom-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Foreslå en tittel: Bystyret vedtok ny gangbro i Trondheim."}]}'
```

Gateway: <http://localhost:8080> · Prometheus: <http://localhost:9090> · Grafana: <http://localhost:3000> (anonymous admin, datasource preconfigured; local demo only).

The response is a normal OpenAI response plus:

```json
"gateway": {"route": "simple", "reason": "short-output task 'title' ...", "latency_s": 0.84,
            "cost_usd": 0.0, "fallbacks": [], "cached": false, "request_id": "..."}
```

`model` accepts `auto`, a route (`simple`, `complex`, `local-only`) or any model name (`local-small`, `claude-smart`, ...). Other endpoints: `GET /v1/usage` (per-team totals, needs a key), `GET /metrics`, `GET /healthz`. Extra request field: `"cache": false` skips the cache.

### Use it from the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="newsroom-dev-key")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Klassifiser denne artikkelen: ..."}],
)
print(resp.choices[0].message.content)
print(resp.model_extra["gateway"])
```

## Newsroom demo

With the gateway running (`pip install -r requirements.txt` once, then):

```bash
python examples/newsroom_demo.py
```

One short Norwegian article, four tasks: tag and headline go to `local-small`, the two-sided summary ("Analyze ...") to `claude-smart`, and the follow-up questions about an unpublished draft to `local-only` (`local-medium`). It prints route, model, latency and cost per task, and the total.

## Running without Docker

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                            # offline, no keys needed

ollama serve &                                       # or point OLLAMA_URL at any OpenAI-compatible server
ollama pull qwen2.5:3b && ollama pull llama3.1:8b
uvicorn --factory gateway.app:create_app --port 8080
```

`GATEWAY_CONFIG` selects another config file (default `./config.yaml`); `OLLAMA_URL` overrides the Ollama address.

## Kubernetes (kind)

```bash
kind create cluster --name llm-gateway
docker build -t llm-gateway:latest .
kind load docker-image llm-gateway:latest --name llm-gateway
kubectl apply -f k8s/ollama.yaml -f k8s/gateway.yaml
kubectl exec deploy/ollama -- ollama pull qwen2.5:3b
kubectl create secret generic gateway-secrets --from-literal=ANTHROPIC_API_KEY=sk-ant-...   # optional
kubectl rollout restart deploy/gateway
kubectl port-forward svc/gateway 8080:80
```

The gateway Deployment runs 2 replicas with probes, resource limits, a non-root user and a read-only root filesystem, and carries Prometheus scrape annotations. See the comments in `k8s/ollama.yaml` for swapping Ollama for vLLM on GPU nodes. Note the in-memory cache and usage counters are **per replica** (see roadmap).

## Evals

```bash
python evals/run_evals.py --gateway http://localhost:8080 --targets local-small claude-fast auto
```

Each of the 12 cases in `evals/cases.jsonl` (summaries in both directions, headline, entity extraction, classification, grounding, "unknown" instead of hallucination, arithmetic, strict JSON, refusing to dox a private person, neutral rewrite) is sent to every target with the cache disabled and scored with `must_contain` / `must_not_contain` / `max_words` (`"a|b"` means either). It prints PASS/FAIL per case, a summary table and writes `evals/results.csv`.

Results (fill in after running on your hardware):

| Target | Pass rate | p50 latency | Total cost |
|---|---|---|---|
| local-small | | | |
| claude-fast | | | |
| auto | | | |

## Design decisions

- **Heuristic router first.** A handful of regexes on the latest user turn, with a reason in every response, is cheap, explainable and testable. A learned router needs labelled traffic you do not have on day one; the evals tell you when the heuristic is leaving quality or money on the table. It is fallible by design: a pasted article containing the word "plan" will route as complex.
- **OpenAI-compatible on both sides.** Clients use any OpenAI SDK unchanged; local backends are anything that speaks `/v1/chat/completions`, so Ollama to vLLM or SGLang is a URL change.
- **Redact only for external providers.** Local models never leave the cluster, so they see the original text and give better answers; anything bound for a hosted API is scrubbed first. The cache key uses the original prompt.
- **Fail open on models, fail closed on guardrails.** A broken model just moves the request to the next one in the chain; a request that trips a guardrail is rejected, never "best-effort" forwarded.
- **Bounded metric labels.** Unknown route names are counted under `route="invalid"` so a client cannot create unbounded label values.
- **Dev keys in config.** API keys live in `config.yaml` for simplicity. For anything real, mount the config from a Secret and rotate the keys. `GET /v1/usage` shows every team's totals to any valid key; restrict that if teams should not see each other's spend.

## Pricing

Anthropic list prices used in `config.yaml` (checked 2026-10-08): Haiku 4.5 $1 / $5 and Sonnet 5.5 $2 / $10 per million input / output tokens. Source: <https://platform.claude.com/docs/en/about-claude/pricing>. Re-check before relying on cost numbers.

## Roadmap

- vLLM vs llama.cpp benchmark on a GPU node (latency, throughput, cost per 1k requests).
- Semantic cache (embedding similarity) on top of the exact-match one.
- Streaming (SSE) responses.
- Redis for the cache and usage counters, so replicas share state.
- Per-team budgets and rate limits, enforced at the gateway.
- LLM-as-judge scoring in the evals for open-ended answers.
