"""Newsroom demo: one short Norwegian article, four tasks, four routing decisions.

Run the gateway first, then:   python examples/newsroom_demo.py

Expected routing with the shipped config:
  1. tag        -> simple     (local-small)
  2. headline   -> simple     (local-small)
  3. two-sided  -> complex    (claude-smart, falls back to local-medium)
  4. private    -> local-only (local-medium) - never leaves the cluster
"""

from __future__ import annotations

import sys

import httpx

from client import chat, text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARTICLE = """\
Bystyret i Trondheim vedtok i går kveld å bygge ny gangbro over Nidelva ved Bakklandet.
Flertallet, bestående av Ap, SV og MDG, mener broen vil gjøre hverdagen tryggere for
syklister og fotgjengere. Prisen er anslått til 85 millioner kroner, og byggingen starter
tidligst våren 2027. Høyre og Frp stemte imot og mener pengene heller bør brukes på
vedlikehold av eksisterende veier. Ordfører Marit Haugen sa etter møtet at «dette er en
investering i en by der flere kan gå og sykle til jobb». Naboer ved Bakklandet har varslet
at de vil klage på vedtaket.
"""

TASKS = [
    ("tag", "Classify this article into one category (politikk, økonomi, sport, kultur, samfunn). "
            f"Reply with the category only.\n\n{ARTICLE}", "auto"),
    ("headline", f"Write one headline for this article, max 10 words.\n\n{ARTICLE}", "auto"),
    ("two-sided summary", "Analyze the arguments for and against the decision and give a "
                          f"balanced two-sided summary.\n\n{ARTICLE}", "auto"),
    ("private follow-up", "This is an unpublished draft. What follow-up questions should I put to the "
                          f"neighbours who plan to appeal?\n\n{ARTICLE}", "local-only"),
]


def main() -> int:
    total = 0.0
    print(f"{'task':<20} {'route':<11} {'model':<14} {'latency':>8} {'cost':>10}")
    print("-" * 68)
    for name, prompt, model in TASKS:
        try:
            resp = chat(prompt, model=model, cache=False)
        except httpx.HTTPError as exc:
            print(f"{name:<20} FAILED: {exc}")
            return 1
        gw = resp["gateway"]
        total += gw["cost_usd"]
        print(f"{name:<20} {gw['route']:<11} {resp['model']:<14} {gw['latency_s']:>7.2f}s ${gw['cost_usd']:>9.6f}")
        print(f"    why: {gw['reason']}")
        if gw["fallbacks"]:
            print(f"    fallbacks: {gw['fallbacks']}")
        print(f"    out: {text(resp)[:160]!r}")
    print("-" * 68)
    print(f"total cost: ${total:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
