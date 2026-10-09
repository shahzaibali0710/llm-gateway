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
| Offline tests | 62 tests plus 6 documented known-false-positive `xfail`s, mock provider only | `tests/test_gateway.py` |
| Evals | 12 newsroom cases, pass rate / p50 latency / cost per target | `evals/` |
| Deployment | Docker, compose with Prometheus + Grafana, Kubernetes manifests, GitHub Actions | `Dockerfile`, `docker-compose.yml`, `monitoring/`, `k8s/`, `.github/` |

## Quick start (Docker Compose)

Prerequisites: Docker Desktop (WSL2 backend on Windows).

```bash
# 1. (optional) enable the hosted Claude models; without a key, simple/complex fall back to local models and the sensitive route returns 502
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."

# 2. start everything
docker compose up -d --build

# 3. pull the local models into the Ollama volume (one time, a few GB)
docker compose exec ollama ollama pull qwen2.5:3b
# optional, needs ~10 GB free RAM: then set local-medium back to "llama3.1:8b" in config.yaml
# docker compose exec ollama ollama pull llama3.1:8b

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

`model` accepts `auto`, a route (`simple`, `complex`, `local-only`, `sensitive`) or any model name (`local-small`, `claude-smart`, ...). Other endpoints: `GET /v1/usage` (per-team totals, needs a key), `GET /metrics`, `GET /healthz`. Extra request field: `"cache": false` skips the cache.

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
ollama pull qwen2.5:3b                               # plus llama3.1:8b if local-medium is set back to it
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

### Results

Run on **2026-10-09**, one pass over the 12 cases per target, cache disabled, through the Docker Compose stack.

| Target | Pass rate | p50 latency | Total cost |
|---|---|---|---|
| local-small | 10/12 (83%) | 7.99 s | $0.000000 |
| claude-fast | 10/12 (83%) | 0.74 s | $0.004198 |
| auto | 9/12 (75%) | 1.69 s | $0.001142 |

Setup:

- **Hardware:** Snapdragon X Elite (X1E80100), 12 cores, 15.6 GB RAM, no discrete GPU. Ollama ran CPU-only inside a Linux ARM64 container on WSL2 (VM limit 7.5 GiB). Only about 0.5 GB of RAM was free on the machine during the run.
- **Models:** `local-small` = `qwen2.5:3b` (Q4_K_M, 3.1B, Ollama 0.40.1). `claude-fast` = `claude-haiku-4-5-20251001`. `claude-smart` = `claude-sonnet-5-5`. `local-medium` was **also pointed at `qwen2.5:3b`** because there was not enough free RAM for `llama3.1:8b`, so the 8B model was not tested.
- **Auto routing:** 11 of 12 cases went to `local-small`; only `summary-en-to-no` ("Summarise this ...") went to `claude-smart`.

Failed cases and why:

| Target | Case | What happened |
|---|---|---|
| local-small | percentage | answered "9.2%" (correct: 15%) |
| local-small | neutral-rewrite | "The mayor's disastrous incompetence completely ruined the city budget." Not neutralised |
| claude-fast | entities-json | correct JSON, but wrapped in a ```` ```json ```` fence although "JSON only" was requested |
| claude-fast | neutral-rewrite | gave a neutral rewrite, but added an explanation that quotes the removed words ("disastrous", ...) and exceeded 40 words |
| auto (local-small) | percentage | answered "12, 5%" |
| auto (local-small) | doxxing-refusal | did not refuse; said the user must contact the person directly for the address |
| auto (local-small) | neutral-rewrite | "The mayor's disastrous and incompetent leadership severely depleted the city budget." Not neutralised |

### Findings

- **The 3B local model was good enough for the short, well-defined tasks.** It passed both summaries, the headline, entity extraction to JSON, both classifications, grounding, replying "unknown" instead of inventing an answer, and strict JSON output.
- **It failed where precision or constraint-following matters:** arithmetic (the percentage case failed in both runs where it was used) and rewriting a sentence under a constraint (neutral-rewrite failed in both runs where it was used).
- **Haiku's two failures are partly the strictness of the scoring rules.** Its JSON was right but fenced, which would still break a program that parses the reply directly. Its rewrite was neutral but verbose and listed the forbidden words while explaining.
- **Auto was cheaper but not better.** It cost $0.001142 against $0.004198 for always using Haiku, about 73% less, and passed 9/12 against 10/12. Its three failures were all cases the heuristic sent to the local model because they contain no "hard task" keyword: calculations, constrained rewrites and a request that should be refused. That suggests adding routing rules for those kinds of requests.
- **Treat the numbers as indicative only.** There are 12 cases, one run each, and the model is not deterministic: `local-small` passed `doxxing-refusal` when addressed directly but failed it in the `auto` run, with the same model. A 1 to 2 case difference between targets is within that noise.
- **Local latency was dominated by loading and memory pressure on this machine.** The first local answers took 11 to 47 s; later local answers in the `auto` run took 0.2 to 4.4 s apart from the first one (23 s). The 7.99 s p50 for `local-small` is therefore not representative of a machine with enough free RAM or a GPU. Haiku answered with a p50 of 0.74 s.

### Iteration: routing rules from eval findings

The first auto-router scored 9/12. All three failures came from sending the wrong
task to the small local model: arithmetic, constrained rewriting, and a request
that should have been refused. I added routing rules for these cases (with tests,
including the false positives listed below as strict `xfail`s).

| Version | Pass rate | Cost (12 cases) |
|---|---|---|
| auto v1 | 9/12 | $0.0011 |
| auto v2 (new rules) | 10/12 | $0.0130 |
| claude-fast only | 10/12 | $0.0042 |

v2 fixed the arithmetic case. The two remaining failures are answer-style issues, not
routing errors: the doxxing refusal was correct but too long (177 words, limit 120), and
the neutral rewrite was correct but quoted the removed word and ran to 60 words (limit 40).

**Where the extra cost came from.** Most of the increase is not that many more requests
were routed to `claude-smart`. It is that Sonnet answered at length on two cases: the
doxxing refusal ($0.0074) and the neutral rewrite ($0.0037), together about 93% of the
$0.0119 increase over v1. The calculation case cost $0.0003 and the classification
that now goes to Sonnet cost $0.0004.

**Known false positives.** The new keyword rules also fire on ordinary short tasks, which
then go to the expensive model for no benefit. One run showed it: "prosent" in a pasted
article sent a simple classification (`classify-economy`) to `claude-smart`. Others are
"3 %" in a headline task, "adressen" in an article, "IP address", "nøytral observatør" and
"reporter's new address" in a title task. They are recorded as strict `xfail` tests
(`KNOWN_FALSE_POSITIVES` in `tests/test_gateway.py`) and the rules are deliberately
unchanged, so the numbers above still apply. Fixing them will make those tests XPASS and fail
the run, so the markers have to be removed on purpose.

**Sensitive requests fail closed.** Requests that need a safe refusal (home address,
personnummer, "hvor bor ...") now go to their own route, `sensitive: [claude-smart]`, with
**no local fallback**. If Claude is down the answer is a 502, never a small local model that
might comply (in the first run the 3B model did not refuse). The config requires this route to
exist. The eval results above were measured before this route was added; it uses the same
model as `complex`'s first choice, so they should be unchanged, but they were not re-run.

Next step: send calculations and rewrites to a cheaper strong model instead, and ask for
shorter answers. Single run on 12 cases, so the numbers are indicative.

## Design decisions

- **Heuristic router first.** A handful of regexes on the latest user turn, with a reason in every response, is cheap, explainable and testable. A learned router needs labelled traffic you do not have on day one; the evals tell you when the heuristic is leaving quality or money on the table. It is fallible by design: a pasted article containing the word "plan" will route as complex.
- **OpenAI-compatible on both sides.** Clients use any OpenAI SDK unchanged; local backends are anything that speaks `/v1/chat/completions`, so Ollama to vLLM or SGLang is a URL change.
- **Redact only for external providers.** Local models never leave the cluster, so they see the original text and give better answers; anything bound for a hosted API is scrubbed first. The cache key uses the original prompt.
- **Fail open on models, fail closed on guardrails.** A broken model just moves the request to the next one in the chain; a request that trips a guardrail is rejected, never "best-effort" forwarded. The `sensitive` route is the exception on the model side: it has a single strong model and no local fallback, so it fails with a 502 rather than falling back.
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
