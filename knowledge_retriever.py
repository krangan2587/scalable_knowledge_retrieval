"""
Open-world knowledge bot -- fully scaled architecture

Implements the pipeline:

  USER QUERY -> Query Analysis -> [Schema-value resolution + Auto-growing
  cluster registry] -> Canonical Query -> Retrieval Layer
  [Dense ANN + Sparse BM25 -> RRF] -> Candidate Set -> Reranker ->
  Top Evidence -> Answer

WHAT IS ACTUALLY GUARANTEED 

  - Two IDENTICAL queries always produce identical output. Trivial, but
    stated because it's the only unconditional guarantee.
  - Two queries that DID join the same auto-cluster share retrieval input
    (the cluster's representative text) and therefore get identical
    results.
  - A query that hits CANONICAL_CACHE never re-executes the retrieval
    layer -- proven by EXECUTION_COUNT below, not asserted.


  - Whether two DIFFERENT but semantically-equivalent phrasings join the
    SAME cluster is NOT guaranteed and IS order-dependent. AutoCanonicalRegistry
    is single-pass nearest-centroid clustering (sometimes called "leader
    clustering"): each new query is compared against the CURRENT state of
    existing cluster centroids, and centroids are running averages that
    change with every join. That makes the join decision for query Y
    depend on what arrived before Y and in what order -- this is an
    inherent, well-known property of this algorithm family, not a bug to
    patch out. Verified empirically: running the same 3 paraphrases
    forward vs. reversed changed which clusters they landed in.
  - So: "any two arbitrary semantically-similar statements are guaranteed
    to collapse to the same canonical query" is FALSE and should not be
    claimed about this system. What's true is narrower: queries that
    happen to cluster together get consistent treatment from then on;
    queries just outside the join threshold get their own cluster and
    won't retroactively merge with a similar one that arrived earlier.
"""

import time
import numpy as np
import spacy
import faiss
from rank_bm25 import BM25Okapi
from itertools import combinations

nlp = spacy.load("en_core_web_md")


def embed(text: str) -> np.ndarray:
    """Swap point: replace with a real sentence-embedding API call in production."""
    vec = nlp(text).vector.astype("float32")
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


DIM = 300


# ---------------------------------------------------------------------------
# QUERY ANALYSIS -- no hand-typed alias registry.
#
# Two genuinely different things were conflated in ENTITY_REGISTRY before:
#   1. Knowing your data has a `transaction_type` column with values like
#      DEBIT/CREDIT -- that's schema introspection, not linguistic
#      hardcoding. Any real system has to know its own field names.
#   2. A hand-typed list of every phrase that might mean DEBIT ("debit",
#      "withdrawal", "outgoing payment"...) -- THIS was the actual
#      hardcoding, and it's what's removed here.
#
# Replacement, built from data instead of typed by hand:
#   A) SCHEMA_VALUES is read directly off the documents already in the KB
#      (kb.docs[i]["metadata"]), not declared anywhere as a constant.
#      Matching a query against a schema value uses embedding similarity
#      to the bare value name itself (zero-shot -- no alias list at all).
#      This is honestly WEAKER than a curated alias table for any single
#      value (a bare string like "DEBIT" is a thinner signal than five
#      hand-picked example phrasings) -- that trade-off is real, not
#      hidden, and is exactly why part B exists.
#   B) AutoCanonicalRegistry: for anything that doesn't match a schema
#      value confidently, queries are clustered dynamically by embedding
#      similarity as they arrive. No categories are declared in advance;
#      clusters spawn from traffic. Two paraphrases land in the same
#      cluster (and therefore the same canonical_key/canonical_text) if
#      they're close enough in embedding space -- this still depends on
#      embedding quality (the same dependency flagged all conversation),
#      but it no longer depends on anyone having anticipated the phrasing.
# ---------------------------------------------------------------------------
BACKGROUND_ANCHORS = [
    "reset my password", "update my email address", "what's the weather today",
    "contact customer support", "change my two factor authentication",
]
SCHEMA_MARGIN = 0.03      # schema-value match must beat background anchors by this
CLUSTER_JOIN_THRESHOLD = 0.90  # how close a query must be to join an existing auto-cluster

_BACKGROUND_VECS = None


def derive_schema_values(kb, axes):
    """Read distinct values for each axis directly off the KB's documents --
    schema introspection, not a hardcoded constant. Returns
    {axis: {value: embed(value)}}."""
    schema = {}
    for axis in axes:
        values = sorted({d["metadata"][axis] for d in kb.docs if axis in d["metadata"]})
        schema[axis] = {v: embed(v) for v in values}
    return schema


def match_schema_value(query_vec, schema_axis_values):
    """Zero-shot match against schema values (bare value strings, no
    aliases). Returns (value, similarity) or (None, best_sim) if it
    doesn't clear the background-anchor margin."""
    global _BACKGROUND_VECS
    if _BACKGROUND_VECS is None:
        _BACKGROUND_VECS = [embed(s) for s in BACKGROUND_ANCHORS]
    best_value, best_sim = None, -1.0
    for value, vec in schema_axis_values.items():
        sim = cos_sim(query_vec, vec)
        if sim > best_sim:
            best_value, best_sim = value, sim
    best_background = max(cos_sim(query_vec, b) for b in _BACKGROUND_VECS)
    if best_sim >= best_background + SCHEMA_MARGIN:
        return best_value, best_sim
    return None, best_sim


def extract_slots(query: str) -> dict:
    """Numeric/comparison slots. Regex on a small, genuinely closed set of
    English comparison words (over/under/more than/less than) -- this is a
    much narrower claim than a domain-vocabulary alias list: comparison
    words are closed-class function words, not open-ended domain terms."""
    import re
    q = query.lower()
    slots = {}
    m = re.search(r"over \$?(\d+)", q) or re.search(r"more than \$?(\d+)", q)
    if m:
        slots["amount_gt"] = int(m.group(1))
    m = re.search(r"under \$?(\d+)", q) or re.search(r"less than \$?(\d+)", q)
    if m:
        slots["amount_lt"] = int(m.group(1))
    m = re.search(r"last (\d+) days?", q)
    if m:
        slots["days"] = int(m.group(1))
    return slots


class AutoCanonicalRegistry:
    """Self-growing canonical-cluster registry. No categories declared in
    advance -- clusters spawn from traffic.

    NOT deterministic in the mathematical sense: which cluster a query
    joins (if any) depends on arrival order, because centroids are running
    averages updated on every join, and join decisions compare against
    whatever the centroid currently is. Two semantically-equivalent
    queries are NOT guaranteed to land in the same cluster -- verified
    empirically (see module docstring). What IS true: once two queries
    HAVE joined the same cluster, they share the cluster's `representative`
    text for retrieval, so their results are identical from that point on."""

    def __init__(self, join_threshold=CLUSTER_JOIN_THRESHOLD):
        self.join_threshold = join_threshold
        self.clusters = []  # list of {id, centroid, representative, count, schema_filter}

    def resolve(self, query: str, query_vec, schema, filterable_axes):
        best_idx, best_sim = None, -1.0
        for i, c in enumerate(self.clusters):
            sim = cos_sim(query_vec, c["centroid"])
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx is not None and best_sim >= self.join_threshold:
            c = self.clusters[best_idx]
            c["centroid"] = (c["centroid"] * c["count"] + query_vec) / (c["count"] + 1)
            c["count"] += 1
            return c["id"], c["representative"], c["schema_filter"], False

        schema_filter = {}
        for axis in filterable_axes:
            value, _ = match_schema_value(query_vec, schema.get(axis, {}))
            if value is not None:
                schema_filter[axis] = value
        new_id = len(self.clusters)
        self.clusters.append({"id": new_id, "centroid": query_vec.copy(),
                               "representative": query, "count": 1, "schema_filter": schema_filter})
        return new_id, query, schema_filter, True


def query_analysis(query: str, registry: "AutoCanonicalRegistry", schema, filterable_axes, tenant_filter=None):
    """Returns (canonical_key, canonical_text, metadata_filter, is_new_cluster)."""
    qvec = embed(query)
    slots = extract_slots(query)
    cluster_id, representative, schema_filter, is_new = registry.resolve(query, qvec, schema, filterable_axes)

    canonical_key = (cluster_id, tuple(sorted(slots.items())))
    canonical_text = representative
    if slots:
        extra = []
        if "amount_gt" in slots:
            extra.append(f"over ${slots['amount_gt']}")
        if "amount_lt" in slots:
            extra.append(f"under ${slots['amount_lt']}")
        if "days" in slots:
            extra.append(f"last {slots['days']} days")
        canonical_text = representative + " " + " ".join(extra)

    metadata_filter = {"tenant": tenant_filter, **schema_filter}
    return canonical_key, canonical_text, metadata_filter, is_new

# ---------------------------------------------------------------------------
# Knowledge base: dense (ANN) index + sparse (BM25) index + metadata store.
# metadata filtering generalized from "tenant only" to arbitrary fields, so
# resolved entity constraints (transaction_type/status) can pre-filter
# retrieval too, not just tenant isolation.
# ---------------------------------------------------------------------------
class ScaledKnowledgeBase:
    def __init__(self):
        self.dense_index = faiss.IndexHNSWFlat(DIM, 32)
        self.dense_index.hnsw.efConstruction = 64
        self.dense_index.hnsw.efSearch = 64
        self.docs = []
        self._bm25 = None
        self._bm25_dirty = False

    def add(self, text: str, metadata: dict):
        doc_id = len(self.docs)
        self.docs.append({"id": doc_id, "text": text, "metadata": metadata})
        vec = embed(text).reshape(1, -1)
        self.dense_index.add(vec)  # true incremental insert -- HNSW supports this natively
        self._bm25_dirty = True    # defer the BM25 rebuild instead of doing it on every add

    def _ensure_bm25(self):
        """rank_bm25 has no incremental-update API -- BM25Okapi's constructor
        recomputes IDF over the whole corpus every time, so a rebuild is
        unavoidable with this library. Lazy + deferred rebuild turns bulk
        loading N docs from O(N^2) (rebuild on every add) into O(N) (one
        rebuild on first search after loading) -- real improvement, but
        still O(N) per rebuild once dirty, and NOT true incremental
        indexing. A production deployment needs a search engine whose
        inverted index supports genuine incremental updates internally
        (Elasticsearch/OpenSearch/Lucene) -- rank_bm25 structurally cannot
        do this, no amount of code around it changes that."""
        if self._bm25_dirty or self._bm25 is None:
            self._bm25 = BM25Okapi([d["text"].lower().split() for d in self.docs])
            self._bm25_dirty = False

    def _metadata_ok(self, doc_id, metadata_filter):
        if not metadata_filter:
            return True
        meta = self.docs[doc_id]["metadata"]
        for k, v in metadata_filter.items():
            if v is None:
                continue
            if meta.get(k) != v:
                return False
        return True

    def dense_search(self, query_vec, k, metadata_filter=None):
        fetch_k = k * 5 if metadata_filter else k
        _, idx = self.dense_index.search(query_vec.reshape(1, -1), min(fetch_k, len(self.docs)))
        return [i for i in idx[0] if i != -1 and self._metadata_ok(i, metadata_filter)][:k]

    def sparse_search(self, query: str, k, metadata_filter=None):
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = np.argsort(-scores)
        return [i for i in ranked if self._metadata_ok(i, metadata_filter)][:k]

    def hybrid_search(self, query: str, query_vec, k=10, metadata_filter=None, rrf_k=60):
        dense_ids = self.dense_search(query_vec, k * 2, metadata_filter)
        sparse_ids = self.sparse_search(query, k * 2, metadata_filter)

        fused_scores = {}
        for rank, doc_id in enumerate(dense_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(fused_scores.items(), key=lambda x: -x[1])
        return [doc_id for doc_id, _ in ranked[:k]]

    def rerank(self, query: str, query_vec, candidate_ids, k=5, reranker_backend=None):
        candidates = [{"id": doc_id, "text": self.docs[doc_id]["text"]} for doc_id in candidate_ids]

        if reranker_backend in ("anthropic", "github_models", "gh_models_cli"):
            try:
                from llm_reranker import llm_rerank, github_models_rerank, gh_models_rerank
                fn = {"anthropic": llm_rerank, "github_models": github_models_rerank,
                      "gh_models_cli": gh_models_rerank}[reranker_backend]
                return fn(query, candidates, top_k=k)
            except Exception as e:
                print(f"[rerank] {reranker_backend} reranker unavailable ({e}); "
                      f"falling back to bi-encoder heuristic for this call.")

        scored = []
        q_terms = set(query.lower().split())
        for doc_id in candidate_ids:
            doc = self.docs[doc_id]
            doc_vec = embed(doc["text"])
            dense_score = float(np.dot(query_vec, doc_vec))
            overlap = len(q_terms & set(doc["text"].lower().split())) / max(len(q_terms), 1)
            combined = 0.7 * dense_score + 0.3 * overlap
            scored.append((doc_id, combined))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def recall_at_k_vs_bruteforce(self, sample_queries, k=5, metadata_filter=None):
        all_vecs = np.stack([embed(d["text"]) for d in self.docs])
        recalls = []
        for q in sample_queries:
            qvec = embed(q)
            sims = all_vecs @ qvec
            eligible = [i for i in range(len(self.docs)) if self._metadata_ok(i, metadata_filter)]
            true_top = sorted(eligible, key=lambda i: -sims[i])[:k]
            ann_top = self.dense_search(qvec, k, metadata_filter)
            recalls.append(len(set(true_top) & set(ann_top)) / k)
        return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# ScaledSemanticCache (fuzzy, threshold-based) has been REMOVED as of this
# version -- AutoCanonicalRegistry supersedes it. Clustering now serves the
# same "collapse near-duplicate queries" role, but produces an inspectable,
# reusable canonical_key/canonical_text instead of an opaque cache hit, and
# applies uniformly to every query instead of only ones that failed to
# resolve against a hardcoded registry.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full pipeline, matching the diagram:
#   Query Analysis -> Canonical Query -> Retrieval Layer (dense+sparse->RRF)
#   -> Candidate Set -> Reranker -> Top Evidence -> Answer
# ---------------------------------------------------------------------------
RETRIEVAL_FLOOR = 0.40
RERANKER_BACKEND = None  # None | "anthropic" | "github_models"
# NOTE: GitHub Models (and the gh_models_cli / github_models backends built
# against it in llm_reranker.py) was PERMANENTLY RETIRED by GitHub on
# July 30, 2026 -- confirmed via github.blog/changelog. It is not a
# temporary outage and will not come back. Those two backends are left in
# llm_reranker.py for reference/history but should not be selected here.
# Use "anthropic" (requires ANTHROPIC_API_KEY) for a working real reranker.

CANONICAL_CACHE = {}    # exact-match on canonical_key -- guarantees single execution for
                         # a repeat/joined key; does NOT guarantee two arbitrary paraphrases
                         # get the same key in the first place (see module docstring)
EXECUTION_COUNT = 0     # proves single execution rather than just claiming it


def _run_retrieval_layer(kb, text_for_retrieval, metadata_filter):
    """The Retrieval Layer box: Dense ANN + Sparse BM25 -> RRF -> Candidate
    Set -> Reranker -> Top Evidence. One shared implementation for both the
    canonical and open-world paths."""
    global EXECUTION_COUNT
    EXECUTION_COUNT += 1

    qvec = embed(text_for_retrieval)
    candidates = kb.hybrid_search(text_for_retrieval, qvec, k=10, metadata_filter=metadata_filter)
    if not candidates:
        return []

    reranked = kb.rerank(text_for_retrieval, qvec, candidates, k=3, reranker_backend=RERANKER_BACKEND)
    reranked = [(doc_id, score) for doc_id, score in reranked if score >= RETRIEVAL_FLOOR]
    return [{"id": doc_id, "text": kb.docs[doc_id]["text"], "score": round(score, 3)} for doc_id, score in reranked]


def answer(kb: ScaledKnowledgeBase, registry: AutoCanonicalRegistry, query: str,
           schema, filterable_axes, tenant_filter=None):
    """Unified path -- AutoCanonicalRegistry always returns a canonical_key
    (either an existing cluster or a freshly spawned one), so there's no
    more open-world/closed-world branch. The old separate fuzzy
    ScaledSemanticCache is superseded: the registry's clustering IS the
    fuzzy-match mechanism now, and it produces an inspectable, reusable
    canonical_key/canonical_text instead of just an opaque cache hit."""
    canonical_key, canonical_text, metadata_filter, is_new = query_analysis(
        query, registry, schema, filterable_axes, tenant_filter)

    cache_key = (canonical_key, tenant_filter)
    if cache_key in CANONICAL_CACHE:
        return {"hits": CANONICAL_CACHE[cache_key], "source": "canonical_cache",
                "canonical_key": canonical_key, "canonical_text": canonical_text,
                "is_new_cluster": is_new, "executed": False}

    hits = _run_retrieval_layer(kb, canonical_text, metadata_filter)
    CANONICAL_CACHE[cache_key] = hits
    note = {} if hits else {"note": "no relevant results found"}
    return {"hits": hits, "source": "retrieval_canonical", "canonical_key": canonical_key,
            "canonical_text": canonical_text, "is_new_cluster": is_new, "executed": True, **note}


# ---------------------------------------------------------------------------
# Demo KB -- now tagged with transaction_type/status metadata so resolved
# entity constraints can actually pre-filter retrieval, not just shape text.
# ---------------------------------------------------------------------------
def build_demo_kb():
    kb = ScaledKnowledgeBase()
    docs = [
        ("Your account has three debit transactions this week totaling $175.",
         {"tenant": "acme", "transaction_type": "DEBIT"}),
        ("You have one large pending withdrawal of $120 flagged for review.",
         {"tenant": "acme", "transaction_type": "DEBIT", "status": "PENDING"}),
        ("A $500 credit deposit was received into your account on the 6th.",
         {"tenant": "acme", "transaction_type": "CREDIT", "status": "COMPLETED"}),
        ("Your account shows one credit transaction that is still pending: $75.",
         {"tenant": "acme", "transaction_type": "CREDIT", "status": "PENDING"}),
        ("Two small debit charges under $50 were completed earlier this month.",
         {"tenant": "acme", "transaction_type": "DEBIT", "status": "COMPLETED"}),
        ("International wire transfers typically take three to five business days.",
         {"tenant": "acme"}),
        ("Our return policy allows refunds within thirty days with a receipt.",
         {"tenant": "acme"}),
        ("Tomorrow's forecast is partly cloudy with a high near seventy-two.",
         {"tenant": "acme"}),
        ("To reset your password, go to account settings and forgot password.",
         {"tenant": "acme"}),
        ("Classic chocolate chip cookies need butter, sugar, flour, baking soda.",
         {"tenant": "acme"}),
        ("Globex account: two debit transactions totaling $60 this week.",
         {"tenant": "globex", "transaction_type": "DEBIT"}),
    ]
    for text, meta in docs:
        kb.add(text, meta)
    return kb


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
CLUSTERS = {
    "all_debits_acme": [
        "give me all debit transactions",
        "show me all withdrawals",
        "what did I spend",
    ],
    "weather": [
        "what's the weather tomorrow",
        "will it be sunny tomorrow",
    ],
    "truly_unanswerable": [
        "what's the capital of Mongolia",
        "who won the 1998 World Cup",
    ],
}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_validation():
    global EXECUTION_COUNT
    kb = build_demo_kb()

    # Filterable axes derived from the data itself -- not a hardcoded list.
    # Any metadata key present on any document except "tenant" is treated
    # as a schema field a query could be resolved against.
    filterable_axes = sorted({k for d in kb.docs for k in d["metadata"] if k != "tenant"})
    schema = derive_schema_values(kb, filterable_axes)
    print(f"Schema derived from KB data (not hardcoded): {filterable_axes} -> "
          f"{ {ax: list(vals.keys()) for ax, vals in schema.items()} }\n")

    print("=== Tenant filtering check ===")
    registry = AutoCanonicalRegistry()
    acme_result = answer(kb, registry, "give me all debit transactions", schema, filterable_axes, tenant_filter="acme")
    globex_result = answer(kb, registry, "give me all debit transactions", schema, filterable_axes, tenant_filter="globex")
    print(f"acme tenant  -> {[h['id'] for h in acme_result['hits']]}")
    print(f"globex tenant-> {[h['id'] for h in globex_result['hits']]}")
    tenant_ok = globex_result['hits'] and all(h['id'] == 10 for h in globex_result['hits'])
    print(f"({'PASS' if tenant_ok else 'REVIEW'}: same phrasing, different tenants -- "
          f"globex must see doc 10, never acme's cached answer)\n")

    print("=== Recall@k monitor (ANN vs brute-force ground truth) ===")
    recall = kb.recall_at_k_vs_bruteforce(["debit transactions", "weather forecast", "cookie recipe"], k=3)
    print(f"recall@3: {recall:.1%}\n")

    print("=== Query Analysis + Canonical Query + Retrieval Layer validation ===")
    for cluster_name, paraphrases in CLUSTERS.items():
        CANONICAL_CACHE.clear()
        EXECUTION_COUNT = 0
        registry = AutoCanonicalRegistry()  # fresh registry per cluster to isolate
        rows = []
        for q in paraphrases:
            result = answer(kb, registry, q, schema, filterable_axes, tenant_filter="acme")
            doc_ids = {h["id"] for h in result["hits"]}
            rows.append({"query": q, "result": result, "doc_ids": doc_ids})

        canonical_keys_seen = {r["result"]["canonical_key"] for r in rows}
        pairs = list(combinations(rows, 2))
        overlaps = [jaccard(a["doc_ids"], b["doc_ids"]) for a, b in pairs] if pairs else [1.0]
        avg_overlap = sum(overlaps) / len(overlaps)

        is_unanswerable = cluster_name == "truly_unanswerable"
        if is_unanswerable:
            status = "PASS" if all(not r["doc_ids"] for r in rows) else "REVIEW"
        else:
            status = "PASS" if avg_overlap == 1.0 and all(r["doc_ids"] for r in rows) else "REVIEW"

        print(f"\n--- {cluster_name} [{status}]  avg overlap: {round(avg_overlap, 3)}  "
              f"distinct canonical keys: {len(canonical_keys_seen)}  "
              f"retrieval-layer executions: {EXECUTION_COUNT} ---")
        for r in rows:
            res = r["result"]
            src = res["source"]
            note = res.get("note", "")
            ck = res.get("canonical_key")
            print(f"  \"{r['query']}\" [{src}] canonical_key={ck} -> "
                  f"doc_ids={sorted(r['doc_ids'])} {note}")


if __name__ == "__main__":
    t0 = time.time()
    run_validation()
    print(f"\n(total run time: {time.time() - t0:.2f}s)")
