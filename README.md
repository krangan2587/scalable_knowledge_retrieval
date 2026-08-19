# Knowledge Retriever

A retrieval pipeline for a knowledge bot: turns a user query into a canonical
form, searches a knowledge base with combined dense (ANN) + sparse (BM25)
search, reranks the results, and returns the top evidence.

```
USER QUERY -> Query Analysis -> Canonical Query -> Retrieval Layer
   [Dense ANN + Sparse BM25 -> RRF] -> Candidate Set -> Reranker -> Answer
```

## Files

| File | Purpose |
|---|---|
| `knowledge_retriever.py` | Main pipeline. Run this. |
| `llm_reranker.py` | Real LLM-based reranker backends. Must sit in the same folder — `knowledge_retriever.py` imports from it. |
| `requirements.txt` | Python dependencies. |

## Install

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_md
```
Add `--break-system-packages` to the `pip install` line if your system requires it for system-wide installs.

## Run

```bash
python3 knowledge_retriever.py
```

This runs a built-in demo/validation: it builds a small sample knowledge base, then checks:
- **Tenant isolation** — different customers never see each other's cached answers.
- **Recall@k** — how accurate the fast approximate search is versus an exact (but slow) search.
- **Query collapsing** — whether differently-worded questions that mean the same thing produce the same answer.

Read the printed output; each section says `PASS` or `REVIEW` and why.

## Enabling a real AI reranker (optional)

By default, the reranker step uses a lightweight fallback and says so in the output (`[rerank] ... unavailable ... falling back`). To use a real model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Then in `knowledge_retriever.py`, set:
```python
RERANKER_BACKEND = "anthropic"
```
and run again. If the fallback message disappears from the output, it's working.

> Note: the `"github_models"` / `"gh_models_cli"` backends in `llm_reranker.py` are **non-functional** — GitHub Models was permanently retired on July 30, 2026. They're kept only as reference code; do not select them.

## Known limitations (read before relying on this)

- The default embedding model is a simple word-vector average, not a modern sentence embedding — this is the main source of imprecision you'll see in the `REVIEW` results.
- Query collapsing uses an auto-growing cluster registry that is **not guaranteed** to group every pair of semantically-equivalent phrasings — it can be order-dependent. See the module docstring in `knowledge_retriever.py` for details.
- This is a demo-scale knowledge base (~11 documents), not a production index.
