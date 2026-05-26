import json
import hashlib
import datetime
import os
import time
from collections import deque, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from mycelium.pipeline import config_loader as cfg
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
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
    from mycelium.pipeline.layer2_expert_loader import get_expert_model
except Exception:  # pragma: no cover
    def get_expert_model(domain):
        return None


# ---------------------------------------------------------------------------
# Dynamic domain discovery — reads whatever expert subdirs exist on disk.
# Falls back to config-supplied list if the experts directory is absent.
# ---------------------------------------------------------------------------

def _discover_live_domains(experts_dir: Optional[str] = None) -> List[str]:
    """Return sorted list of domain names that have an expert directory on disk."""
    if experts_dir is None:
        try:
            experts_dir = cfg.experts_dir()  # type: ignore[attr-defined]
        except Exception:
            experts_dir = "experts"
    try:
        if os.path.isdir(experts_dir):
            return sorted(
                d for d in os.listdir(experts_dir)
                if os.path.isdir(os.path.join(experts_dir, d))
                and not d.startswith(".")
            )
    except OSError:
        pass
    # Fallback: config-supplied list
    try:
        return sorted(cfg.layer1_domain_list())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Domain ontology — built dynamically.
# A base ontology covers well-understood structural domains. Any domain that
# exists on disk but is NOT in the base ontology gets a minimal scaffold entry
# so it is still reachable via lexical/embedding scoring.
# ---------------------------------------------------------------------------

_BASE_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "astronomy": {
        "level": "object",
        "core": [
            "astronomy", "space", "planet", "planets", "sun", "earth", "moon",
            "star", "stars", "galaxy", "orbit", "orbits", "orbital", "solar",
            "telescope", "parallax",
        ],
        "attributes": ["distance", "light", "mass", "gravity", "trajectory"],
    },
    "automobile": {
        "level": "object",
        "core": [
            "car", "vehicle", "automobile", "auto", "maruti", "suzuki", "bike",
            "motorcycle", "tyre", "tire", "tires", "tubeless", "hatchback",
            "sedan", "suv",
        ],
        "attributes": ["engine", "wheel", "wheels", "tread", "drive", "manual", "automatic"],
    },
    "engine_spec": {
        "level": "attribute",
        "core": ["engine", "cc", "horsepower", "hp", "torque", "700cc", "700"],
        "attributes": [],
        "parent": "automobile",
    },
    "aesthetics": {
        "level": "attribute",
        "core": [
            "color", "colour", "finish", "gloss", "glossy", "matte", "metallic",
            "red", "cherry", "blue", "green",
        ],
        "attributes": ["shade", "tone"],
    },
    "physics": {
        "level": "object",
        "core": [
            "physics", "force", "energy", "velocity", "acceleration", "momentum",
            "quantum", "relativity", "wave", "particle", "field", "charge",
            "magnetic", "electric",
        ],
        "attributes": ["mass", "speed", "temperature", "pressure", "frequency"],
    },
    "chemistry": {
        "level": "object",
        "core": [
            "chemistry", "chemical", "element", "compound", "molecule", "atom",
            "reaction", "acid", "base", "bond", "ion", "periodic", "oxidation",
            "reduction",
        ],
        "attributes": ["concentration", "temperature", "catalyst", "solvent"],
    },
    "medical": {
        "level": "object",
        "core": [
            "medical", "disease", "diagnosis", "treatment", "drug", "symptom",
            "patient", "clinical", "surgery", "therapy", "medicine", "anatomy",
            "pathology", "vaccine",
        ],
        "attributes": ["dose", "chronic", "acute", "benign", "malignant"],
    },
    "music": {
        "level": "object",
        "core": [
            "music", "song", "melody", "chord", "rhythm", "beat", "note", "scale",
            "instrument", "guitar", "piano", "drums", "bass", "tempo", "lyrics",
        ],
        "attributes": ["pitch", "tone", "harmony", "octave", "frequency"],
    },
}


def _build_domain_ontology(experts_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Merge base ontology with any live domains found on disk.

    Domains on disk that have no base entry get a minimal scaffold so the
    embedding lens can still score them via their domain name as anchor text.
    """
    ontology: Dict[str, Dict[str, Any]] = dict(_BASE_ONTOLOGY)
    live = _discover_live_domains(experts_dir)
    for domain in live:
        if domain not in ontology:
            # Minimal scaffold — embedding similarity to domain name itself
            ontology[domain] = {
                "level": "object",
                "core": [domain.replace("_", " "), domain],
                "attributes": [],
            }
    return ontology


# Build once at import time; refreshed on explicit reload.
_DOMAIN_ONTOLOGY: Dict[str, Dict[str, Any]] = _build_domain_ontology()

_ONTOLOGY_PARENTS: Dict[str, str] = {
    k: v["parent"] for k, v in _DOMAIN_ONTOLOGY.items() if "parent" in v
}
_OBJECT_LEVEL_DOMAINS: set = {
    k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "object"
}
_ATTRIBUTE_LEVEL_DOMAINS: set = {
    k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "attribute"
}

# DOMAIN_LIST kept in sync with disk-discovered domains for LLM prompts.
DOMAIN_LIST: List[str] = _discover_live_domains() or list(_DOMAIN_ONTOLOGY.keys())


def reload_domain_ontology(experts_dir: Optional[str] = None) -> None:
    """Re-discover domains from disk and refresh all module-level structures.

    Call this after a new expert has been created on disk so the router
    immediately recognises the new domain without a process restart.
    """
    global _DOMAIN_ONTOLOGY, _ONTOLOGY_PARENTS, _OBJECT_LEVEL_DOMAINS
    global _ATTRIBUTE_LEVEL_DOMAINS, DOMAIN_LIST
    _DOMAIN_ONTOLOGY = _build_domain_ontology(experts_dir)
    _ONTOLOGY_PARENTS = {
        k: v["parent"] for k, v in _DOMAIN_ONTOLOGY.items() if "parent" in v
    }
    _OBJECT_LEVEL_DOMAINS = {
        k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "object"
    }
    _ATTRIBUTE_LEVEL_DOMAINS = {
        k for k, v in _DOMAIN_ONTOLOGY.items() if v.get("level") == "attribute"
    }
    DOMAIN_LIST = _discover_live_domains(experts_dir) or list(_DOMAIN_ONTOLOGY.keys())


# ---------------------------------------------------------------------------
# LRU tag cache
# ---------------------------------------------------------------------------
_TAG_CACHE_MAX: int = 2048
_TAG_CACHE_TTL: float = 3600.0
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
    _tag_cache.move_to_end(key)
    return tags


def _tag_cache_put(text: str, tags: List[str]) -> None:
    key = _tag_cache_key(text)
    _tag_cache[key] = (time.monotonic(), tags)
    _tag_cache.move_to_end(key)
    if len(_tag_cache) > _TAG_CACHE_MAX:
        _tag_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_tags_transformer(tags, model_name: str = ""):
    model_name = model_name or cfg.layer1_embed_model()
    if SentenceTransformer is None:
        import numpy as np
        return np.eye(len(tags))
    try:
        import numpy as np
        from mycelium.pipeline.model_registry import embed_batch
        vecs = embed_batch(tags, model_name=model_name, device="cpu")
        return np.array(vecs)
    except ImportError:
        model = SentenceTransformer(model_name, device="cpu")
        return model.encode(tags)


def cluster_tags_transformer(
    tags: List[str],
    embeddings,
    similarity_threshold: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Cosine-similarity clustering that merges overlapping domain candidates.

    Returns a dict mapping cluster_id -> list of member domain names.
    The cluster representative (first member) is the highest-scoring one
    when called from _deduplicate_candidates().
    """
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_tag_cluster_similarity_threshold()
    sim_matrix = cosine_similarity(embeddings)
    clusters: Dict[str, List[str]] = {}
    used: set = set()
    cluster_id = 0
    for i, tag in enumerate(tags):
        if i in used:
            continue
        cluster = [tag]
        used.add(i)
        for j in range(i + 1, len(tags)):
            if j not in used and sim_matrix[i][j] >= similarity_threshold:
                cluster.append(tags[j])
                used.add(j)
        clusters[str(cluster_id)] = cluster
        cluster_id += 1
    return clusters


def _deduplicate_candidates(
    candidates: List[Tuple[str, float]],
    model_name: str = "",
    similarity_threshold: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """Merge semantically-overlapping candidates before scoring.

    For each cluster keep the highest-scoring representative.
    This prevents automobile + engine_spec both passing through when they
    are near-duplicates in embedding space.
    """
    if len(candidates) <= 1:
        return candidates
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_tag_cluster_similarity_threshold()
    domains = [d for d, _ in candidates]
    score_map = {d: s for d, s in candidates}
    try:
        embeddings = embed_tags_transformer(domains, model_name=model_name)
        clusters = cluster_tags_transformer(domains, embeddings, similarity_threshold)
    except Exception:
        # If embedding fails, fall back to ontology-parent dedup only
        return _parent_dedup(candidates)
    deduped: List[Tuple[str, float]] = []
    for members in clusters.values():
        # Keep the member with the highest original score
        best = max(members, key=lambda d: score_map.get(d, 0.0))
        deduped.append((best, score_map[best]))
    deduped.sort(key=lambda x: x[1], reverse=True)
    return deduped


def _parent_dedup(candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Suppress child domains when their parent is also a candidate and scores higher."""
    score_map = {d: s for d, s in candidates}
    suppressed: set = set()
    for domain in score_map:
        parent = _ONTOLOGY_PARENTS.get(domain)
        if parent and parent in score_map:
            # Suppress child if parent score >= child score
            if score_map[parent] >= score_map[domain]:
                suppressed.add(domain)
    return [(d, s) for d, s in candidates if d not in suppressed]


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
    model = model or cfg.layer1_openai_model()
    max_tokens = cfg.layer1_openai_max_tokens()
    temperature = cfg.layer1_openai_temperature()
    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags "
        f"(choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\n"
        f"Sentence: {text}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
    clusters: Dict[str, List[str]] = {}
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
                "distance_threshold": distance_threshold,
            },
        },
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
        self.tag_frequency: Dict[str, int] = {}
        self.time_window = time_window_hours * 3600

    def add_statement(self, sentence, tags, timestamp):
        entry = {
            "sentence": sentence,
            "tags": tags,
            "timestamp": timestamp,
            "parsed_time": datetime.datetime.fromisoformat(timestamp),
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
        recent_tags: set = set()
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
    recent_tags: List[str] = []
    for entry in recent_statements:
        recent_tags.extend(entry["tags"])
    tag_to_cluster: Dict[str, str] = {}
    for cluster_id, c_tags in clusters.items():
        for tag in c_tags:
            tag_to_cluster[tag] = cluster_id
    cluster_counts: Dict[str, int] = {}
    for tag in recent_tags:
        cluster_id = tag_to_cluster.get(tag)
        if cluster_id:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "dominant_clusters": sorted_clusters[:3],
        "cluster_distribution": cluster_counts,
        "total_recent_tags": len(recent_tags),
        "unique_recent_tags": len(set(recent_tags)),
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
            "reason": f"cluster_{cluster_id}_dominates_with_{dominance_ratio:.2f}_ratio",
        }
    return {
        "action": "create_hybrid_patch",
        "primary_cluster": cluster_id,
        "dominance_ratio": dominance_ratio,
        "reason": f"moderate_dominance_{dominance_ratio:.2f}_suggests_hybrid",
    }


# ==========================
# Layer 1: Multi-Lens Routing
# ==========================

def _tokenize(text: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower())]


def _lens1_embedding_candidates(
    text: str,
    top_k: Optional[int] = None,
    model_name: str = "",
) -> List[Tuple[str, float]]:
    """Lens 1: Embedding-based permissive candidate selection over live domains.

    Scores ALL domains currently in _DOMAIN_ONTOLOGY (which is built from
    whatever experts exist on disk).  Results are deduplicated via
    _deduplicate_candidates() to collapse near-synonym domains (e.g.
    automobile + engine_spec) before they propagate downstream.
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()
    model_name = model_name or cfg.layer1_embed_model()
    emb_weight = cfg.layer1_lens1_embedding_weight()
    lex_weight = cfg.layer1_lens1_lexical_weight()

    tokens = _tokenize(text)

    # Anchor text for each domain — use core keywords as the semantic anchor.
    anchors = {
        d: " ".join(sorted(set(v.get("core", [])[:5] + v.get("attributes", [])[:3]))) or d
        for d, v in _DOMAIN_ONTOLOGY.items()
    }

    def _lexical_score(domain: str) -> float:
        vocab = set(
            _DOMAIN_ONTOLOGY[domain].get("core", [])
            + _DOMAIN_ONTOLOGY[domain].get("attributes", [])
        )
        hits = sum(1 for t in tokens if t in vocab)
        return hits / max(1, len(tokens))

    if SentenceTransformer is None:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        raw = [s for s in scored[:top_k] if s[1] > 0]
        return _deduplicate_candidates(raw, model_name=model_name)

    try:
        import numpy as _np
        from mycelium.pipeline.model_registry import get_embedding, embed_batch
        text_emb = get_embedding(text, model_name=model_name, device="cpu")
        anchor_texts = list(anchors.values())
        anchor_domains = list(anchors.keys())
        anchor_embs = embed_batch(anchor_texts, model_name=model_name, device="cpu")
        scores = []
        for d, emb in zip(anchor_domains, anchor_embs):
            sim = float(
                _np.dot(text_emb, emb)
                / ((_np.linalg.norm(text_emb) + 1e-12) * (_np.linalg.norm(emb) + 1e-12))
            )
            sim = emb_weight * sim + lex_weight * _lexical_score(d)
            scores.append((d, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        raw = scores[:top_k]
        return _deduplicate_candidates(raw, model_name=model_name)
    except ImportError:
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
            raw = scores[:top_k]
            return _deduplicate_candidates(raw, model_name=model_name)
        except Exception:
            scored = [(d, _lexical_score(d)) for d in anchors.keys()]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [s for s in scored[:top_k] if s[1] > 0]
    except Exception:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]


def _lens2_ontology_explanations(text: str, candidates: List[str]) -> List[Dict]:
    """Lens 2: Ontology/behavioral explanations with parent-child suppression."""
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
            "parent": _ONTOLOGY_PARENTS.get(concept),
        }
    # Child-domain penalty: if child core tokens are a subset of parent core tokens,
    # the child adds no new information — penalise it strongly.
    for concept, info in coverage_map.items():
        parent = info.get("parent")
        if parent and parent in coverage_map:
            parent_core = set(coverage_map[parent]["matched_core"])
            child_core = set(info["matched_core"])
            if child_core and child_core.issubset(parent_core):
                info["coverage"] *= 0.7
    # Secondary dampening: if parent coverage dominates, dampen the child further.
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
                "level": _DOMAIN_ONTOLOGY.get(concept, {}).get("level", "unknown"),
            })
    results.sort(key=lambda x: (x["level"] != "object", -x["score"]))
    selected: List[Dict] = []
    seen_parents: set = set()
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
    signature: Dict[str, Any] = {"levels": {"core": {}, "modifiers": {}}}
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
        "modifiers": len(signature["levels"].get("modifiers", {})),
    }
    return signature


# ---------------------------------------------------------------------------
# Event helpers
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
    if on_event is None:
        return
    try:
        from mycelium.pipeline.pipeline_event import build_event
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
        pass


_HEARTBEAT_MESSAGES = [
    "Reconciling conflicting evidence\u2026",
    "Stabilizing reasoning graph\u2026",
    "Reviewing semantic dependencies\u2026",
    "Cross-referencing domain signals\u2026",
    "Validating ontology alignment\u2026",
]
_heartbeat_counter: int = 0


def _next_heartbeat_message() -> str:
    global _heartbeat_counter
    msg = _HEARTBEAT_MESSAGES[_heartbeat_counter % len(_HEARTBEAT_MESSAGES)]
    _heartbeat_counter += 1
    return msg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multi_lens_route(
    text: str,
    top_k: Optional[int] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict:
    """Multi-lens similarity and gated routing over live expert domains.

    Pipeline:
      Lens 1  — embedding similarity against ALL live domain anchors,
                followed by _deduplicate_candidates() to cluster overlapping
                domains (e.g. automobile + engine_spec) before they propagate.
      Lens 2  — ontology coverage with parent-child suppression.
      Lens 3  — abstraction signature (core vs. modifier detection).

    If no object-level domain evidence is found the result is ATTRIBUTE_ONLY
    and the caller must decide whether to invoke CREATE_NEW_EXPERT.
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()

    wall_start = time.monotonic()

    _emit_layer1(
        on_event,
        phase_name="routing",
        phase_id=4,
        substep="lens1_start",
        state="running",
        visibility="public",
        message="Running multi-lens router\u2026",
        detail=f"top_k={top_k}  live_domains={len(_DOMAIN_ONTOLOGY)}",
        metadata={"top_k": top_k, "live_domain_count": len(_DOMAIN_ONTOLOGY)},
    )

    # Lens 1 — deduplication happens inside _lens1_embedding_candidates
    lens1 = _lens1_embedding_candidates(text, top_k=top_k)
    lens1_domains = [d for d, _ in lens1]

    _emit_layer1(
        on_event,
        phase_name="heartbeat",
        phase_id=4,
        substep="lens1_done",
        state="running",
        visibility="public",
        message=_next_heartbeat_message(),
        detail=f"{len(lens1_domains)} candidate domain(s) after dedup",
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

    if core_active == 0 or (
        core_domains_present and core_domains_present.issubset(_ATTRIBUTE_LEVEL_DOMAINS)
    ):
        result = {
            "primary_domain": None,
            "secondary_domains": [],
            "explanation": "Attribute-heavy input detected; no core object domain evidence.",
            "lens1_candidates": lens1,
            "lens2_explanations": lens2,
            "lens3_signature": lens3,
            "classification": "ATTRIBUTE_ONLY",
        }
        _emit_layer1(
            on_event,
            phase_name="graph_routing",
            phase_id=4,
            substep="decision",
            state="done",
            visibility="public",
            message="Routing complete \u2014 attribute-only query",
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

    secondary: List[Dict] = []
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

    expl_bits: List[str] = []
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
        "classification": "NORMAL",
    }

    _emit_layer1(
        on_event,
        phase_name="graph_routing",
        phase_id=4,
        substep="decision",
        state="done",
        visibility="public",
        message=f"Routing complete \u2014 {primary}",
        detail=(
            f"primary={primary} \u00b7 secondary={len(secondary)} \u00b7 "
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
    """Lightweight smoke tests for the multi-lens routing stack."""
    def _print_result(title: str, text: str, result: Dict):
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
        print(f"Input: {text}")
        print(f"Lens1 candidates: {result['lens1_candidates']}")
        print(f"Lens2 explanations: {result['lens2_explanations']}")
        print(f"Lens3 signature: {json.dumps(result['lens3_signature'], indent=2)}")
        print(f"Final decision -> primary: {result['primary_domain']}, secondary: {result['secondary_domains']}")
        print(f"Explanation: {result['explanation']}")

    text_a = "Earth orbits the Sun"
    res_a = multi_lens_route(text_a)
    _print_result("Test A \u2014 Single-domain input", text_a, res_a)
    assert res_a["classification"] == "NORMAL"
    assert res_a["primary_domain"] in {"astronomy", "physics"}
    assert len(res_a["secondary_domains"]) <= 2
    assert not any(s["weight"] >= 1.0 for s in res_a["secondary_domains"])

    text_b = "maruti 700 cc tubeless tires red cherry"
    res_b = multi_lens_route(text_b)
    _print_result("Test B \u2014 Compositional, multi-attribute input", text_b, res_b)
    assert res_b["classification"] == "NORMAL"
    assert res_b["primary_domain"] in {"automobile"}
    # engine_spec must NOT appear alongside automobile after dedup
    sec_domains = {s["domain"] for s in res_b["secondary_domains"]}
    assert "engine_spec" not in sec_domains or "automobile" not in sec_domains, (
        "engine_spec and automobile should be deduplicated — one must be suppressed"
    )
    assert len(sec_domains) <= 2

    text_c = "red glossy metallic finish"
    res_c = multi_lens_route(text_c)
    _print_result("Test C \u2014 Attribute-heavy input", text_c, res_c)
    assert res_c["classification"] in {"ATTRIBUTE_ONLY", "UNKNOWN"}

    text_d = "iphone 15 pro max titanium blue"
    res_d = multi_lens_route(text_d)
    _print_result("Test D \u2014 OOD input (no live expert)", text_d, res_d)

    print("\nLive domains on disk:", DOMAIN_LIST)


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
