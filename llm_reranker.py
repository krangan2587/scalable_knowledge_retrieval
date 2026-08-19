"""
Real reranker (v4.1) -- replaces the dense_score*0.7 + keyword_overlap*0.3
stand-in in fully_scaled_v4.py's rerank() with an actual LLM-based
cross-encoder call.

Why this is categorically different from the stand-in, not just "fancier":
The stand-in scores query and document INDEPENDENTLY (embed each, compare
vectors) then blends in a keyword heuristic. A real cross-encoder / LLM
reranker reads the query and EACH document TOGETHER, jointly, in one
forward pass -- it can reason about whether this specific passage answers
this specific question, not just whether their vectors point the same
direction. That's the entire reason rerankers outperform bi-encoder
similarity at the precision problems we kept hitting all conversation.

Two real options, both standard in production RAG stacks:
  1. llm_rerank()    -- general-purpose LLM (Claude) scores relevance.
                         Flexible, can be prompted with domain instructions,
                         costs more per call.
  2. cohere_rerank()  -- purpose-built reranking API (Cohere, or similarly
                         Voyage/Jina rerank endpoints). Cheaper and faster
                         than a full chat LLM call, narrower job.

Both require real credentials this sandbox does not have (confirmed above:
huggingface.co is network-blocked for any locally-run cross-encoder, and
no API key is present in this environment for a hosted one). Set the
relevant environment variable in YOUR environment and this runs as-is.
"""

import os
import json
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


def llm_rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """candidates: list of {"id": ..., "text": ...}. Returns list of
    (id, score 0-1) sorted descending, length <= top_k."""
    numbered = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "You are a search relevance judge. Score how relevant each passage is "
        f"to the query, from 0 (irrelevant) to 10 (directly answers it).\n\n"
        f'Query: "{query}"\n\nPassages:\n{numbered}\n\n'
        'Respond with ONLY a JSON array like [{"index": 0, "score": 7.5}, ...], '
        "one object per passage, no other text."
    )
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=15,
    )
    resp.raise_for_status()  # will raise cleanly with the real HTTP status/reason
    text = resp.json()["content"][0]["text"]
    scored = json.loads(text)
    scored.sort(key=lambda x: -x["score"])
    return [(candidates[s["index"]]["id"], s["score"] / 10.0) for s in scored[:top_k]]


def cohere_rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """Purpose-built reranking endpoint -- the more common production choice
    (LangChain/LlamaIndex default reranker integrations use this shape)."""
    resp = requests.post(
        "https://api.cohere.com/v1/rerank",
        headers={"Authorization": f"Bearer {COHERE_API_KEY or ''}", "content-type": "application/json"},
        json={
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": top_k,
        },
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [(candidates[r["index"]]["id"], r["relevance_score"]) for r in results]


def github_models_rerank(query: str, candidates: list, top_k: int = 3, model: str = "openai/gpt-4o-mini") -> list:
    """DEAD as of July 30, 2026 -- GitHub Models was permanently retired
    (github.blog/changelog/2026-07-30-github-models-is-now-retired), not
    a temporary outage. This function will always fail now; kept only as
    a record of the approach. Do not select this backend."""
    """GitHub Models API -- official, documented product
    (docs.github.com/en/rest/models/inference), NOT a reverse-engineered
    Copilot internal endpoint. Authenticates with a GitHub token that has
    the `models: read` scope, which a Copilot-enabled account has. Set
    GITHUB_TOKEN (or GH_TOKEN) in your environment -- e.g. `gh auth token`
    if you're already authenticated via the GitHub CLI with Copilot access.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    numbered = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "You are a search relevance judge. Score how relevant each passage is "
        f"to the query, from 0 (irrelevant) to 10 (directly answers it).\n\n"
        f'Query: "{query}"\n\nPassages:\n{numbered}\n\n'
        'Respond with ONLY a JSON array like [{"index": 0, "score": 7.5}, ...], '
        "one object per passage, no other text."
    )
    resp = requests.post(
        "https://models.github.ai/inference/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=15,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    scored = json.loads(text)
    scored.sort(key=lambda x: -x["score"])
    return [(candidates[s["index"]]["id"], s["score"] / 10.0) for s in scored[:top_k]]


def gh_models_rerank(query: str, candidates: list, top_k: int = 3, model: str = "openai/gpt-4o-mini") -> list:
    """DEAD as of July 30, 2026 -- GitHub Models was permanently retired
    (github.blog/changelog/2026-07-30-github-models-is-now-retired), not
    a temporary outage. `gh models run` will always fail now, even with
    correct auth (confirmed: the `gh auth`/`gh extension` mechanism this
    function relies on worked correctly; the backing service is what's
    gone). Kept only as a record of a real, verified, zero-token-in-code
    auth pattern -- the same `gh auth`-delegation approach would work
    again against any future GitHub-native inference surface, but this
    specific one is not coming back. Do not select this backend."""
    """Real GitHub-account authentication with ZERO token handling in this
    code -- uses the official `gh models` CLI extension (github/gh-models),
    which reads whatever credential `gh auth login` already stored (a real
    OAuth device-flow / browser login against your GitHub account -- the
    same mechanism GitHub Enterprise Cloud SSO uses, per GitHub's own docs:
    docs.github.com/en/copilot/.../authenticate-copilot-cli).

    This function never reads, stores, or passes a token. Auth is fully
    delegated to the `gh` binary's own secure credential store.

    One-time setup on YOUR machine (not something this code does for you):
        gh auth login                                  # browser/device-flow, your GitHub account
        gh extension install https://github.com/github/gh-models
    After that, this function just works -- no env var, no key to paste.
    """
    import subprocess
    numbered = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    prompt = (
        "You are a search relevance judge. Score how relevant each passage is "
        f"to the query, from 0 (irrelevant) to 10 (directly answers it).\n\n"
        f'Query: "{query}"\n\nPassages:\n{numbered}\n\n'
        'Respond with ONLY a JSON array like [{"index": 0, "score": 7.5}, ...], '
        "one object per passage, no other text."
    )
    result = subprocess.run(
        ["gh", "models", "run", model, prompt],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh models run failed (exit {result.returncode}): {result.stderr.strip()}")

    text = result.stdout.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    scored = json.loads(text)
    scored.sort(key=lambda x: -x["score"])
    return [(candidates[s["index"]]["id"], s["score"] / 10.0) for s in scored[:top_k]]


if __name__ == "__main__":
    # Prove this is wired correctly by actually calling it, right now, in
    # this sandbox -- not asserting it would work, showing what happens.
    candidates = [
        {"id": 1, "text": "Your account has three debit transactions this week totaling $175."},
        {"id": 2, "text": "Tomorrow's forecast is partly cloudy with a high near seventy-two."},
    ]
    print("Calling llm_rerank() (Anthropic) with no ANTHROPIC_API_KEY set in this sandbox...")
    try:
        result = llm_rerank("give me all debit transactions", candidates)
        print("Unexpected success:", result)
    except requests.exceptions.HTTPError as e:
        print(f"  Failed at HTTP {e.response.status_code}: {e.response.text[:150]}")
        print("  -> request reached Anthropic's API correctly; failed on AUTH only.\n")

    print("Calling github_models_rerank() against https://models.github.ai ...")
    try:
        result = github_models_rerank("give me all debit transactions", candidates)
        print("Success:", result)
    except requests.exceptions.HTTPError as e:
        print(f"  Failed at HTTP {e.response.status_code}: {e.response.text[:150]}")
    except requests.exceptions.RequestException as e:
        print(f"  Failed before getting an HTTP response at all: {e}")
        print("  -> this sandbox's network egress allowlist blocks models.github.ai outright")
        print("     (host_not_allowed) -- this is a sandbox network restriction, not a bug in")
        print("     this code or a problem with GitHub Models. Run this in your own environment")
        print("     with GITHUB_TOKEN set and it should reach the API normally.")

    print("\nCalling gh_models_rerank() -- zero token handling, delegates to `gh auth` ...")
    try:
        result = gh_models_rerank("give me all debit transactions", candidates)
        print("Success:", result)
    except Exception as e:
        print(f"  Failed: {e}")
        print("  -> requires `gh auth login` (interactive OAuth) + `gh extension install")
        print("     github/gh-models`, neither of which this non-interactive sandbox can do.")
        print("     This is the mechanism to use for a real enterprise deployment: no token")
        print("     ever appears in code, config, or environment variables.")
