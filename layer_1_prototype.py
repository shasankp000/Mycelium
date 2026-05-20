import json
import hashlib
import datetime
import time
from collections import deque, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
import config_loader as cfg
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - allow graceful degradation
    SentenceTransformer = None  # type: ignore
# Cosine similarity with graceful fallback if scikit-learn is unavailable
try:
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity  # type: ignore
    def cosine_similarity(a, b=None):
        return _sk_cosine_similarity(a, b)
except Exception:  # pragma: no cover
    import math as _math
    def _ensure_2d(x):
        if not x:
            return [[0.0]]
        if isinstance(x[0], (list, tuple)):
            return [[float(v) for v in row] for row in x]
        return [[float(v) for v in x]]

    def _dot(u, v):
        return sum((ui * vi for ui, vi in zip(u, v)))

    def _norm(u):
        return _math.sqrt(sum((ui * ui for ui in u))) + 1e-12

    def cosine_similarity(a, b=None):
        A = _ensure_2d(a)
        B = _ensure_2d(b) if b is not None else A
        out = []
        for u in A:
            row = []
            un = _norm(u)
            for v in B:
                vn = _norm(v)
                row.append(_dot(u, v) / (un * vn))
            out.append(row)
        return out
try:
    from fuzzywuzzy import process
except Exception:  # pragma: no cover
    import difflib as _difflib
    class _ProcessStub:
        @staticmethod
        def extractOne(query, choices):
            best = None
            best_score = -1
            for c in choices:
                score = int(100 * _difflib.SequenceMatcher(None, str(query).lower(), str(c).lower()).ratio())
                if score > best_score:
                    best = c
                    best_score = score
            return best, best_score
    process = _ProcessStub()
try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:  # pragma: no cover
    class AgglomerativeClustering:
        def __init__(self, n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3):
            self.labels_ = []
        def fit(self, embeddings):
            self.labels_ = list(range(len(embeddings)))
            return self
try:
    import ollama
except Exception:  # pragma: no cover
    class _OllamaStub:
        @staticmethod
        def generate(model: str, prompt: str):
            return {"response": ""}
    ollama = _OllamaStub()
try:
    from layer_2_prototype import get_expert_model
except Exception:  # pragma: no cover
    def get_expert_model(domain):
        return None

# ---------------------------------------------------------------------------
# Domain list — loaded from config; fallback to built-in default
# ---------------------------------------------------------------------------
DOMAIN_LIST: List[str] = cfg.layer1_domain_list()


# ---------------------------------------------------------------------------
# LLM tag extraction cache (§4.1)
# — Keyed by sha256(text).  Each entry is (timestamp_float, tags_list).
# — Max 2048 entries, LRU eviction, 1-hour TTL.
# ---------------------------------------------------------------------------
_TAG_CACHE_MAX: int = 2048
_TAG_CACHE_TTL: float = 3600.0          # seconds
_tag_cache: "OrderedDict[str, Tuple[float, List[str]]]" = OrderedDict()


def _tag_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tag_cache_get(text: str) -> Optional[List[str]]:
    key = _tag_cache_key(text)
    if key not in _tag_cache:
        return None
    ts, tags = _tag_cache[key]
    if time.monotonic() - ts > _TAG_CACHE_TTL:
        del _tag_cache[key]
        return None
    _tag_cache.move_to_end(key)          # LRU refresh
    return tags


def _tag_cache_put(text: str, tags: List[str]) -> None:
    key = _tag_cache_key(text)
    _tag_cache[key] = (time.monotonic(), tags)
    _tag_cache.move_to_end(key)
    if len(_tag_cache) > _TAG_CACHE_MAX:
        _tag_cache.popitem(last=False)   # evict oldest


# ---------------------------------------------------------------------------
# embed_tags_transformer — uses model_registry to avoid duplicate loads (§1.1)
# ---------------------------------------------------------------------------

def embed_tags_transformer(tags, model_name: str = ""):
    """Embed *tags* using SentenceTransformer on CPU.

    Delegates to model_registry.get_model() so the encoder is loaded once
    per process and served from the in-memory cache on every subsequent call.
    Also uses model_registry.embed_batch() so individual tag embeddings that
    were already computed earlier in the same request are returned as pure
    cache hits with no encode() call.
    """
    model_name = model_name or cfg.layer1_embed_model()
    if SentenceTransformer is None:
        import numpy as np
        embeddings = np.eye(len(tags))
        return embeddings
    try:
        import numpy as np
        from model_registry import embed_batch
        vecs = embed_batch(tags, model_name=model_name, device="cpu")
        return np.array(vecs)
    except ImportError:
        # model_registry not available — fall back to direct load
        model = SentenceTransformer(model_name, device="cpu")
        return model.encode(tags)

def cluster_tags_transformer(tags, embeddings, similarity_threshold: Optional[float] = None):
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_tag_cluster_similarity_threshold()
    sim_matrix = cosine_similarity(embeddings)
    clusters = {}
    used = set()
    cluster_id = 0
    for i, tag in enumerate(tags):
        if i in used:
            continue
        cluster = [tag]
        used.add(i)
        for j in range(i+1, len(tags)):
            if sim_matrix[i][j] >= similarity_threshold and j not in used:
                cluster.append(tags[j])
                used.add(j)
        clusters[str(cluster_id)] = cluster
        cluster_id += 1
    return clusters

def extract_tags_openai(
    text,
    api_key,
    endpoint: str = "",
    provider: str = "",
    model: str = "",
):
    import requests
    endpoint = endpoint or cfg.layer1_openai_endpoint()
    provider = provider or cfg.layer1_openai_provider()
    model    = model    or cfg.layer1_openai_model()
    max_tokens  = cfg.layer1_openai_max_tokens()
    temperature = cfg.layer1_openai_temperature()

    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags "
        f"(choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\n"
        f"Sentence: {text}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    if provider in ("openai", "openrouter", "perplexity"):
        tags_str = result["choices"][0]["message"]["content"]
    else:
        tags_str = result.get("response", "")
    tags_lines = tags_str.splitlines()
    tags_clean = []
    for line in tags_lines:
        if any(domain in line for domain in DOMAIN_LIST):
            tags_clean.extend([tag.strip() for tag in line.split(',') if tag.strip()])
    if not tags_clean:
        tags_clean = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    return tags_clean


def extract_tags_llama(text, model: str = ""):
    """Extract domain tags via Ollama with an LRU + TTL result cache (§4.1).

    Cache key: sha256(text).  Max 2048 entries, 1-hour TTL, LRU eviction.
    Identical or recently-seen sentences are served entirely from memory
    with zero Ollama round-trips.
    """
    # Fast path: exact-match cache hit
    cached = _tag_cache_get(text)
    if cached is not None:
        return cached

    model = model or cfg.layer1_llama_model()
    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags "
        f"(choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\n"
        f"Sentence: {text}"
    )
    response = ollama.generate(model=model, prompt=prompt)
    tags_str = response['response']
    tags_lines = tags_str.splitlines()
    tags_clean = []
    for line in tags_lines:
        if any(domain in line for domain in DOMAIN_LIST):
            tags_clean.extend([tag.strip() for tag in line.split(',') if tag.strip()])
    if not tags_clean:
        tags_clean = [tag.strip() for tag in tags_str.split(',') if tag.strip()]

    _tag_cache_put(text, tags_clean)
    return tags_clean

def normalize_tags(tags, domain_list=None, threshold: Optional[int] = None):
    if domain_list is None:
        domain_list = DOMAIN_LIST
    if threshold is None:
        threshold = cfg.layer1_tag_normalize_threshold()
    normalized = []
    for tag in tags:
        match, score = process.extractOne(tag, domain_list)
        normalized.append(match if score >= threshold else tag)
    return normalized

def cluster_tags(tags, embeddings, distance_threshold: Optional[float] = None):
    if distance_threshold is None:
        distance_threshold = cfg.layer1_tag_cluster_distance_threshold()
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    ).fit(embeddings)
    clusters = {}
    for tag, label in zip(tags, clustering.labels_):
        clusters.setdefault(str(label), []).append(tag)
    return clusters

def save_clusters_to_json(clusters, text, extractor, embed_model, distance_threshold, filename: str = ""):
    filename = filename or cfg.layer1_tag_clusters_filename()
    data = {
        "clusters": clusters,
        "metadata": {
            "input_text": text,
            "extractor": extractor,
            "embedding_model": embed_model,
            "clustering": {
                "method": "AgglomerativeClustering",
                "linkage": "average",
                "distance_threshold": distance_threshold
            }
        }
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_clusters_from_json(filename: str = ""):
    filename = filename or cfg.layer1_tag_clusters_filename()
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# Temporal Locality Layer
class TemporalLocalityLayer:
    def __init__(
        self,
        max_size: Optional[int] = None,
        time_window_hours: Optional[float] = None,
    ):
        if max_size is None:
            max_size = cfg.layer1_temporal_max_size()
        if time_window_hours is None:
            time_window_hours = cfg.layer1_temporal_time_window_hours()
        self.recent_statements = deque(maxlen=max_size)
        self.tag_frequency = {}
        self.time_window = time_window_hours * 3600

    def add_statement(self, sentence, tags, timestamp):
        entry = {
            "sentence": sentence,
            "tags": tags,
            "timestamp": timestamp,
            "parsed_time": datetime.datetime.fromisoformat(timestamp)
        }
        self.recent_statements.append(entry)
        for tag in tags:
            self.tag_frequency[tag] = self.tag_frequency.get(tag, 0) + 1

    def get_recent_statements(self, time_limit_hours: Optional[float] = None):
        if time_limit_hours is None:
            time_limit_hours = cfg.layer1_temporal_window_hours()
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=time_limit_hours)
        return [entry for entry in self.recent_statements if entry["parsed_time"] >= cutoff_time]

    def get_temporal_similarity(self, input_tags, time_limit_hours: Optional[float] = None):
        if time_limit_hours is None:
            time_limit_hours = cfg.layer1_temporal_window_hours()
        recent_statements = self.get_recent_statements(time_limit_hours)
        if not recent_statements:
            return 0.0
        recent_tags = set()
        for entry in recent_statements:
            recent_tags.update(entry["tags"])
        input_set = set(input_tags)
        if not recent_tags or not input_set:
            return 0.0
        intersection = len(input_set.intersection(recent_tags))
        union = len(input_set.union(recent_tags))
        return intersection / union if union > 0 else 0.0

    def get_frequent_tags(self, min_frequency=2):
        return {tag: freq for tag, freq in self.tag_frequency.items() if freq >= min_frequency}

def analyze_spatial_locality(recent_statements, clusters):
    if not recent_statements:
        return {"dominant_clusters": [], "cluster_distribution": {}}
    recent_tags = []
    for entry in recent_statements:
        recent_tags.extend(entry["tags"])
    tag_to_cluster = {}
    for cluster_id, cluster_tags in clusters.items():
        for tag in cluster_tags:
            tag_to_cluster[tag] = cluster_id
    cluster_counts = {}
    for tag in recent_tags:
        cluster_id = tag_to_cluster.get(tag)
        if cluster_id:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "dominant_clusters": sorted_clusters[:3],
        "cluster_distribution": cluster_counts,
        "total_recent_tags": len(recent_tags),
        "unique_recent_tags": len(set(recent_tags))
    }

def assign_domain_patch(spatial_analysis, similarity_threshold: Optional[float] = None):
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_spatial_dominance_threshold()
    if not spatial_analysis["dominant_clusters"]:
        return {"action": "create_new_patch", "reason": "no_dominant_clusters"}
    dominant_cluster = spatial_analysis["dominant_clusters"][0]
    cluster_id, frequency = dominant_cluster
    total_tags = spatial_analysis["total_recent_tags"]
    dominance_ratio = frequency / total_tags if total_tags > 0 else 0
    if dominance_ratio > similarity_threshold:
        return {
            "action": "use_existing_patch",
            "cluster_id": cluster_id,
            "dominance_ratio": dominance_ratio,
            "reason": f"cluster_{cluster_id}_dominates_with_{dominance_ratio:.2f}_ratio"
        }
    else:
        return {
            "action": "create_hybrid_patch",
            "primary_cluster": cluster_id,
            "dominance_ratio": dominance_ratio,
            "reason": f"moderate_dominance_{dominance_ratio:.2f}_suggests_hybrid"
        }


# ==========================
# Layer 1: Multi-Lens Routing
# ==========================

_DOMAIN_ONTOLOGY: Dict[str, Dict[str, List[str]]] = {
    "astronomy": {
        "level": "object",
        "core": [
            "astronomy", "space", "planet", "planets", "sun", "earth", "moon", "star", "stars", "galaxy", "orbit", "orbits", "orbital", "solar", "telescope", "parallax"
        ],
        "attributes": ["distance", "light", "mass", "gravity", "trajectory"],
    },
    "automobile": {
        "level": "object",
        "core": [
            "car", "vehicle", "automobile", "auto", "maruti", "suzuki", "bike", "motorcycle", "tyre", "tire", "tires", "tubeless", "hatchback", "sedan", "suv"
        ],
        "attributes": ["engine", "wheel", "wheels", "tread", "drive", "manual", "automatic"],
    },
    "engine_spec": {
        "level": "attribute",
        "core": ["engine", "cc", "horsepower", "hp", "torque", "700cc", "700"],
        "attributes": [],
        "parent": "automobile"
    },
    "aesthetics": {
        "level": "attribute",
        "core": [
            "color", "colour", "finish", "gloss", "glossy", "matte", "metallic", "red", "cherry", "blue", "green"
        ],
        "attributes": ["shade", "tone"]
    },
}

_ONTOLOGY_PARENTS: Dict[str, str] = {k: v["parent"] for k, v in _DOMAIN_ONTOLOGY.items() if "parent" in v}
_OBJECT_LEVEL_DOMAINS: set = {k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "object"}
_ATTRIBUTE_LEVEL_DOMAINS: set = {k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "attribute"}

def _tokenize(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower())]

def _lens1_embedding_candidates(
    text: str,
    top_k: Optional[int] = None,
    model_name: str = "",
) -> List[Tuple[str, float]]:
    """Lens 1: Embedding-based permissive candidate selection.

    Uses model_registry.get_embedding() + embed_batch() so the query and
    anchor embeddings are computed once and cached — subsequent calls with
    the same text/anchors are pure in-memory cache hits (§1.1).
    Gracefully degrades to lexical scoring if sentence-transformers or
    model_registry are unavailable.
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()
    model_name = model_name or cfg.layer1_embed_model()
    emb_weight = cfg.layer1_lens1_embedding_weight()
    lex_weight = cfg.layer1_lens1_lexical_weight()

    tokens = _tokenize(text)

    anchors = {
        d: " ".join(sorted(set(v.get("core", [])[:5] + v.get("attributes", [])[:3]))) or d
        for d, v in _DOMAIN_ONTOLOGY.items()
    }

    def _lexical_score(domain: str) -> float:
        vocab = set(_DOMAIN_ONTOLOGY[domain].get("core", []) + _DOMAIN_ONTOLOGY[domain].get("attributes", []))
        hits = sum(1 for t in tokens if t in vocab)
        return hits / max(1, len(tokens))

    if SentenceTransformer is None:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]

    # Embedding scoring — uses registry cache to avoid duplicate encode() calls
    try:
    	import numpy as _np
        from model_registry import get_embedding, embed_batch
        text_emb = get_embedding(text, model_name=model_name, device="cpu")
        anchor_texts = list(anchors.values())
        anchor_domains = list(anchors.keys())
        anchor_embs = embed_batch(anchor_texts, model_name=model_name, device="cpu")
        scores = []
        for d, emb in zip(anchor_domains, anchor_embs):
            sim = float(
                _np.dot(text_emb, emb) / ((_np.linalg.norm(text_emb) + 1e-12) * (_np.linalg.norm(emb) + 1e-12))
            )
            sim = emb_weight * sim + lex_weight * _lexical_score(d)
            scores.append((d, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    except ImportError:
        # model_registry not available — fall back to direct SentenceTransformer load
        try:
            model = SentenceTransformer(model_name, device="cpu")
            text_emb = model.encode([text])[0]
            dom_embs = {d: model.encode([a])[0] for d, a in anchors.items()}
            scores = []
            for d, emb in dom_embs.items():
                sim = float(cosine_similarity([text_emb], [emb])[0][0])
                sim = emb_weight * sim + lex_weight * _lexical_score(d)
                scores.append((d, sim))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception:
            scored = [(d, _lexical_score(d)) for d in anchors.keys()]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s in scored[:top_k] if s[1] > 0]
    except Exception:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]


def _lens2_ontology_explanations(text: str, candidates: List[str]) -> List[Dict]:
    """Lens 2: Ontology/behavioral explanations."""
    tokens = set(_tokenize(text))
    results = []
    coverage_map: Dict[str, Dict] = {}
    for concept in candidates:
        ont = _DOMAIN_ONTOLOGY.get(concept, {})
        core = set(ont.get("core", []))
        attrs = set(ont.get("attributes", []))
        matched_core = sorted(list(tokens & core))
        matched_attr = sorted(list(tokens & attrs))
        coverage = len(matched_core) + 0.5 * len(matched_attr)
        coverage_map[concept] = {
            "coverage": coverage,
            "matched_core": matched_core,
            "matched_attr": matched_attr,
            "is_child": concept in _ONTOLOGY_PARENTS,
            "parent": _ONTOLOGY_PARENTS.get(concept)
        }
    for concept, info in coverage_map.items():
        parent = info.get("parent")
        if parent and parent in coverage_map:
            parent_core = set(coverage_map[parent]["matched_core"]) if coverage_map[parent] else set()
            child_core = set(info["matched_core"]) if info else set()
            if child_core and child_core.issubset(parent_core):
                info["coverage"] *= 0.7
    for concept, info in coverage_map.items():
        parent = info.get("parent")
        if parent and parent in coverage_map:
            parent_cov = coverage_map[parent]["coverage"]
            if parent_cov >= 0.9 * info["coverage"]:
                info["coverage"] *= 0.9
    for concept, info in coverage_map.items():
        if info["coverage"] > 0:
            results.append({
                "concept": concept,
                "score": round(float(info["coverage"]), 4),
                "matched_core": info["matched_core"],
                "matched_attr": info["matched_attr"],
                "parent": info.get("parent"),
                "level": _DOMAIN_ONTOLOGY.get(concept, {}).get("level", "unknown")
            })
    results.sort(key=lambda x: (x["level"] != "object", -x["score"]))
    selected = []
    seen_parents = set()
    for r in results:
        p = r.get("parent")
        if p and any(s["concept"] == p for s in results):
            parent_score = next(s["score"] for s in results if s["concept"] == p)
            if parent_score >= 0.9 * r["score"]:
                continue
        if r["concept"] in seen_parents:
            continue
        selected.append(r)
        seen_parents.add(r["concept"])
    return selected[:3]


def _lens3_abstraction_signature(text: str, concepts: List[str]) -> Dict:
    """Lens 3: Abstraction/superposition detection."""
    tokens = _tokenize(text)
    signature = {"levels": {"core": {}, "modifiers": {}}}
    for c in concepts:
        ont = _DOMAIN_ONTOLOGY.get(c, {})
        core_hits = [t for t in tokens if t in set(ont.get("core", []))]
        mod_hits = [t for t in tokens if t in set(ont.get("attributes", []))]
        if core_hits:
            signature["levels"]["core"][c] = core_hits
        if mod_hits:
            signature["levels"]["modifiers"][c] = mod_hits
    signature["active_domains"] = {
        "core": len(signature["levels"].get("core", {})),
        "modifiers": len(signature["levels"].get("modifiers", {}))
    }
    return signature


# ---------------------------------------------------------------------------
# _emit_layer1 — shared event helper for multi_lens_route
# ---------------------------------------------------------------------------

def _emit_layer1(
    on_event: Optional[Callable[[Dict[str, Any]], None]],
    phase_name: str,
    phase_id: int,
    substep: str,
    state: str,
    visibility: str,
    message: str,
    detail: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire a PipelineEvent through *on_event* if registered.

    Exceptions from the callback are caught and silently discarded so a
    broken telemetry path can never abort the routing pipeline.
    """
    if on_event is None:
        return
    try:
        from pipeline_event import build_event
        ev = build_event(
            phase_name=phase_name,
            phase_id=phase_id,
            substep=substep,
            state=state,
            visibility=visibility,
            message=message,
            detail=detail,
            metadata=metadata or {},
        )
        on_event(ev)
    except Exception:
        pass  # telemetry must never crash the pipeline


# ---------------------------------------------------------------------------
# Semantic heartbeat — emitted mid-routing to signal liveness (§9)
# ---------------------------------------------------------------------------

# Human-readable continuity messages rotated round-robin so repeated
# heartbeats don't look identical on the frontend.
_HEARTBEAT_MESSAGES = [
    "Reconciling conflicting evidence…",
    "Stabilizing reasoning graph…",
    "Reviewing semantic dependencies…",
    "Cross-referencing domain signals…",
    "Validating ontology alignment…",
]
_heartbeat_counter: int = 0


def _next_heartbeat_message() -> str:
    global _heartbeat_counter
    msg = _HEARTBEAT_MESSAGES[_heartbeat_counter % len(_HEARTBEAT_MESSAGES)]
    _heartbeat_counter += 1
    return msg


def multi_lens_route(
    text: str,
    top_k: Optional[int] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict:
    """Public API: Multi-lens similarity and gated routing.

    Stage 5 (transparency emitter): accepts an optional ``on_event``
    callback.  Three PipelineEvents are emitted:

    routing (running, public, phase_id=4)
        — fired immediately when routing begins so the frontend can show
          an active indicator as soon as multi-lens work starts.

    heartbeat (running, public, phase_id=4)
        — fired after Lens 1 (the heaviest step, involves embeddings)
          completes, before Lens 2/3 begin.  Acts as a semantic
          continuity signal (§9) — not an exact internal dump.

    graph_routing (done, public, phase_id=4)
        — fired when all three lenses have resolved and the final
          routing decision is assembled.  Carries the primary domain
          and classification in metadata.
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()

    wall_start = time.monotonic()

    # ── routing: started ───────────────────────────────────────────────
    _emit_layer1(
        on_event,
        phase_name="routing",
        phase_id=4,
        substep="lens1_start",
        state="running",
        visibility="public",
        message="Running multi-lens router…",
        detail=f"top_k={top_k}",
        metadata={"top_k": top_k},
    )

    lens1 = _lens1_embedding_candidates(text, top_k=top_k)
    lens1_domains = [d for d, _ in lens1]

    # ── heartbeat: after Lens 1 (embedding step) ───────────────────────
    _emit_layer1(
        on_event,
        phase_name="heartbeat",
        phase_id=4,
        substep="lens1_done",
        state="running",
        visibility="public",
        message=_next_heartbeat_message(),
        detail=f"{len(lens1_domains)} candidate domain(s) after embedding pass",
        metadata={
            "candidate_count": len(lens1_domains),
            "elapsed_ms": round((time.monotonic() - wall_start) * 1000, 1),
        },
    )

    lens2 = _lens2_ontology_explanations(text, lens1_domains or list(_DOMAIN_ONTOLOGY.keys()))
    considered = lens1_domains or [e["concept"] for e in lens2]
    lens3 = _lens3_abstraction_signature(text, considered)

    core_active = lens3["active_domains"]["core"]
    core_domains_present = set(lens3["levels"].get("core", {}).keys())

    if core_active == 0 or (core_domains_present and core_domains_present.issubset(_ATTRIBUTE_LEVEL_DOMAINS)):
        result = {
            "primary_domain": None,
            "secondary_domains": [],
            "explanation": "Attribute-heavy input detected; no core object domain evidence.",
            "lens1_candidates": lens1,
            "lens2_explanations": lens2,
            "lens3_signature": lens3,
            "classification": "ATTRIBUTE_ONLY"
        }
        _emit_layer1(
            on_event,
            phase_name="graph_routing",
            phase_id=4,
            substep="decision",
            state="done",
            visibility="public",
            message="Routing complete — attribute-only query",
            detail="No core object domain evidence detected",
            metadata={
                "primary_domain": None,
                "classification": "ATTRIBUTE_ONLY",
                "elapsed_ms": round((time.monotonic() - wall_start) * 1000, 1),
            },
        )
        return result

    object_level_explanations = [e for e in lens2 if e["concept"] in _OBJECT_LEVEL_DOMAINS]
    attribute_level_explanations = [e for e in lens2 if e["concept"] in _ATTRIBUTE_LEVEL_DOMAINS]

    if object_level_explanations:
        primary = object_level_explanations[0]["concept"]
        primary_score = object_level_explanations[0]["score"]
        remaining_explanations = object_level_explanations[1:] + attribute_level_explanations
    elif lens2:
        primary = lens2[0]["concept"]
        primary_score = lens2[0]["score"]
        remaining_explanations = lens2[1:]
    else:
        primary = lens1[0][0] if lens1 else "unknown"
        primary_score = lens1[0][1] if lens1 else 0.0
        remaining_explanations = []

    secondary = []
    for e in remaining_explanations:
        parent = _ONTOLOGY_PARENTS.get(e["concept"]) if e else None
        if parent == primary:
            continue
        weight = min(0.6, max(0.2, e["score"] / max(1.0, primary_score + 1e-6) * 0.6))
        if weight > 0.2:
            secondary.append({"domain": e["concept"], "weight": round(float(weight), 3)})
        if len(secondary) >= 2:
            break

    modifiers_map = lens3["levels"].get("modifiers", {})
    for c, hits in modifiers_map.items():
        if c != primary and all(sd["domain"] != c for sd in secondary):
            secondary.append({"domain": c, "weight": 0.25})
            break

    secondary = secondary[:2]

    expl_bits = []
    if lens2:
        top = next((e for e in lens2 if e.get("concept") == primary), lens2[0])
        if top.get("matched_core"):
            expl_bits.append(f"Selected {primary} due to core features: {', '.join(top['matched_core'])}.")
        if top.get("matched_attr"):
            expl_bits.append(f"Attributes seen: {', '.join(top['matched_attr'])}.")
    else:
        expl_bits.append(f"Selected {primary} via embedding similarity fallback.")

    if secondary:
        sec_txt = ", ".join([f"{s['domain']} (w={s['weight']})" for s in secondary])
        expl_bits.append(f"Secondary signals detected: {sec_txt}.")

    explanation = " ".join(expl_bits) or "Routing based on multi-lens analysis."

    result = {
        "primary_domain": primary,
        "secondary_domains": secondary,
        "explanation": explanation,
        "lens1_candidates": lens1,
        "lens2_explanations": lens2,
        "lens3_signature": lens3,
        "classification": "NORMAL"
    }

    # ── graph_routing: done ────────────────────────────────────────────
    _emit_layer1(
        on_event,
        phase_name="graph_routing",
        phase_id=4,
        substep="decision",
        state="done",
        visibility="public",
        message=f"Routing complete — {primary}",
        detail=(
            f"primary={primary} · secondary={len(secondary)} · "
            f"classification=NORMAL"
        ),
        metadata={
            "primary_domain": primary,
            "secondary_domains": [s["domain"] for s in secondary],
            "classification": "NORMAL",
            "elapsed_ms": round((time.monotonic() - wall_start) * 1000, 1),
        },
    )

    return result


def test_layer1_multilens() -> None:
    """Lightweight tests for the multi-lens routing stack."""
    def _print_result(title: str, text: str, result: Dict):
        print("\n" + "="*90)
        print(title)
        print("="*90)
        print(f"Input: {text}")
        print(f"Lens1 candidates: {result['lens1_candidates']}")
        print(f"Lens2 explanations: {result['lens2_explanations']}")
        print(f"Lens3 signature: {json.dumps(result['lens3_signature'], indent=2)}")
        print(f"Final decision -> primary: {result['primary_domain']}, secondary: {result['secondary_domains']}")
        print(f"Explanation: {result['explanation']}")

    text_a = "Earth orbits the Sun"
    res_a = multi_lens_route(text_a)
    _print_result("Test A — Single-domain input", text_a, res_a)
    assert res_a["classification"] == "NORMAL"
    assert res_a["primary_domain"] in {"astronomy", "physics"}
    assert len(res_a["secondary_domains"]) <= 2
    assert not any(s["weight"] >= 1.0 for s in res_a["secondary_domains"])

    text_b = "maruti 700 cc tubeless tires red cherry"
    res_b = multi_lens_route(text_b)
    _print_result("Test B — Compositional, multi-attribute input", text_b, res_b)
    assert res_b["classification"] == "NORMAL"
    assert res_b["primary_domain"] in {"automobile"}
    sec_domains = {s["domain"] for s in res_b["secondary_domains"]}
    assert len(sec_domains) <= 2
    assert all(s["weight"] < 1.0 for s in res_b["secondary_domains"])

    text_c = "red glossy metallic finish"
    res_c = multi_lens_route(text_c)
    _print_result("Test C — Attribute-heavy input", text_c, res_c)
    assert res_c["classification"] in {"ATTRIBUTE_ONLY", "UNKNOWN"}
    if res_c["classification"] == "ATTRIBUTE_ONLY":
        assert res_c["primary_domain"] is None
        assert len(res_c["secondary_domains"]) == 0
    for res in (res_a, res_b, res_c):
        total_domains = 1 + len(res["secondary_domains"]) if res["primary_domain"] else len(res["secondary_domains"])
        assert total_domains <= 3

    text_d = "iphone 15 pro max titanium blue"
    res_d = multi_lens_route(text_d)
    _print_result("Test D — Custom input", text_d, res_d)

if __name__ == "__main__":
    print("Running Layer 1 multi-lens routing tests...")
    test_layer1_multilens()
    print("\nAll Layer 1 multi-lens routing tests completed.")

    print("\nQuick manual test:")
    while True:
        text = input("Enter text (or 'exit'): ")
        if text.lower() == "exit":
            break
        res = multi_lens_route(text)
        print(json.dumps(res, indent=2))
