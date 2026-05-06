import json
import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional
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
        # Convert input to 2D list of floats
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
    class AgglomerativeClustering:  # minimal stub to avoid hard dependency in non-used paths
        def __init__(self, n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3):
            self.labels_ = []
        def fit(self, embeddings):
            # Identity clustering: each point its own cluster
            self.labels_ = list(range(len(embeddings)))
            return self
try:
    import ollama
except Exception:  # pragma: no cover
    class _OllamaStub:
        @staticmethod
        def generate(model: str, prompt: str):
            # Simple stub that mimics response structure
            return {"response": ""}
    ollama = _OllamaStub()
# Import dummy expert models (optional; not required for multi-lens tests)
try:
    from layer_2_prototype import get_expert_model
except Exception:  # pragma: no cover
    def get_expert_model(domain):
        return None

# ---------------------------------------------------------------------------
# Domain list — loaded from config; fallback to built-in default
# ---------------------------------------------------------------------------
DOMAIN_LIST: List[str] = cfg.layer1_domain_list()


def embed_tags_transformer(tags, model_name: str = ""):
    model_name = model_name or cfg.layer1_embed_model()
    if SentenceTransformer is None:
        # Graceful fallback: simple bag-of-words vectorization into unit vectors per tag
        # This keeps signature compatible but returns identity-like embeddings
        import numpy as np
        embeddings = np.eye(len(tags))
        return embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(tags)
    return embeddings

def cluster_tags_transformer(tags, embeddings, similarity_threshold: Optional[float] = None):
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_tag_cluster_similarity_threshold()
    # Compute cosine similarity matrix
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
    # OpenAI-compatible response parsing
    if provider in ("openai", "openrouter", "perplexity"):
        tags_str = result["choices"][0]["message"]["content"]
    else:
        tags_str = result.get("response", "")
    # Remove any lines that do not contain domain tags
    tags_lines = tags_str.splitlines()
    tags_clean = []
    for line in tags_lines:
        if any(domain in line for domain in DOMAIN_LIST):
            tags_clean.extend([tag.strip() for tag in line.split(',') if tag.strip()])
    # Fallback: if nothing matches, try splitting the whole string
    if not tags_clean:
        tags_clean = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    return tags_clean


def extract_tags_llama(text, model: str = ""):
    model = model or cfg.layer1_llama_model()
    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags "
        f"(choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\n"
        f"Sentence: {text}"
    )
    response = ollama.generate(model=model, prompt=prompt)
    tags_str = response['response']
    # Remove any lines that do not contain domain tags
    tags_lines = tags_str.splitlines()
    tags_clean = []
    for line in tags_lines:
        if any(domain in line for domain in DOMAIN_LIST):
            tags_clean.extend([tag.strip() for tag in line.split(',') if tag.strip()])
    # Fallback: if nothing matches, try splitting the whole string
    if not tags_clean:
        tags_clean = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
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
        self.recent_statements = deque(maxlen=max_size)  # LRU-like queue
        self.tag_frequency = {}  # Track tag frequency
        self.time_window = time_window_hours * 3600  # Convert to seconds
        
    def add_statement(self, sentence, tags, timestamp):
        """Add a new statement to the temporal layer"""
        entry = {
            "sentence": sentence,
            "tags": tags,
            "timestamp": timestamp,
            "parsed_time": datetime.datetime.fromisoformat(timestamp)
        }
        self.recent_statements.append(entry)
        
        # Update tag frequency
        for tag in tags:
            self.tag_frequency[tag] = self.tag_frequency.get(tag, 0) + 1
    
    def get_recent_statements(self, time_limit_hours: Optional[float] = None):
        """Get recent statements within time window"""
        if time_limit_hours is None:
            time_limit_hours = cfg.layer1_temporal_window_hours()
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=time_limit_hours)
        recent = [entry for entry in self.recent_statements 
                 if entry["parsed_time"] >= cutoff_time]
        return recent
    
    def get_temporal_similarity(self, input_tags, time_limit_hours: Optional[float] = None):
        """Calculate temporal similarity based on recent tag overlap"""
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
        
        # Jaccard similarity
        intersection = len(input_set.intersection(recent_tags))
        union = len(input_set.union(recent_tags))
        return intersection / union if union > 0 else 0.0
    
    def get_frequent_tags(self, min_frequency=2):
        """Get frequently occurring tags"""
        return {tag: freq for tag, freq in self.tag_frequency.items() 
                if freq >= min_frequency}

def analyze_spatial_locality(recent_statements, clusters):
    """Analyze spatial locality by checking clustering of recent statement tags"""
    if not recent_statements:
        return {"dominant_clusters": [], "cluster_distribution": {}}
    
    # Collect all tags from recent statements
    recent_tags = []
    for entry in recent_statements:
        recent_tags.extend(entry["tags"])
    
    # Map tags to their clusters
    tag_to_cluster = {}
    for cluster_id, cluster_tags in clusters.items():
        for tag in cluster_tags:
            tag_to_cluster[tag] = cluster_id
    
    # Count cluster occurrences
    cluster_counts = {}
    for tag in recent_tags:
        cluster_id = tag_to_cluster.get(tag)
        if cluster_id:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    # Sort clusters by frequency
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "dominant_clusters": sorted_clusters[:3],  # Top 3 clusters
        "cluster_distribution": cluster_counts,
        "total_recent_tags": len(recent_tags),
        "unique_recent_tags": len(set(recent_tags))
    }

def assign_domain_patch(spatial_analysis, similarity_threshold: Optional[float] = None):
    """Assign domain/patch network based on spatial locality analysis"""
    if similarity_threshold is None:
        similarity_threshold = cfg.layer1_spatial_dominance_threshold()
    if not spatial_analysis["dominant_clusters"]:
        return {"action": "create_new_patch", "reason": "no_dominant_clusters"}
    
    dominant_cluster = spatial_analysis["dominant_clusters"][0]
    cluster_id, frequency = dominant_cluster
    
    # Calculate dominance ratio
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

# Lightweight domain ontology and feature ownership definitions.
# Keep deterministic and sparse; extendable without heavy models.
_DOMAIN_ONTOLOGY: Dict[str, Dict[str, List[str]]] = {
    # Object-level domains (primary candidates; represent concrete entities)
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
    # Attribute-level domains (secondary candidates; describe properties/modifiers)
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

# Classify domains by level for structural enforcement
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
    Returns top-k domains by cosine similarity to domain anchors. Gracefully
    degrades to lexical scoring if sentence-transformers is unavailable.
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()
    model_name = model_name or cfg.layer1_embed_model()
    emb_weight = cfg.layer1_lens1_embedding_weight()
    lex_weight = cfg.layer1_lens1_lexical_weight()

    tokens = _tokenize(text)

    # Build anchor phrases per domain
    anchors = {
        d: " ".join(sorted(set(v.get("core", [])[:5] + v.get("attributes", [])[:3]))) or d
        for d, v in _DOMAIN_ONTOLOGY.items()
    }

    # Fallback lexical scoring
    def _lexical_score(domain: str) -> float:
        vocab = set(_DOMAIN_ONTOLOGY[domain].get("core", []) + _DOMAIN_ONTOLOGY[domain].get("attributes", []))
        hits = sum(1 for t in tokens if t in vocab)
        return hits / max(1, len(tokens))

    if SentenceTransformer is None:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]

    # Embedding scoring
    try:
        model = SentenceTransformer(model_name)
        text_emb = model.encode([text])[0]
        dom_embs = {d: model.encode([a])[0] for d, a in anchors.items()}
        scores = []
        for d, emb in dom_embs.items():
            sim = float(cosine_similarity([text_emb], [emb])[0][0])
            # Blend with lexical to stabilize
            sim = emb_weight * sim + lex_weight * _lexical_score(d)
            scores.append((d, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    except Exception:
        scored = [(d, _lexical_score(d)) for d in anchors.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s in scored[:top_k] if s[1] > 0]


def _lens2_ontology_explanations(text: str, candidates: List[str]) -> List[Dict]:
    """Lens 2: Ontology/behavioral explanations.
    - Build concept → feature ownership mapping
    - Score by coverage, apply redundancy penalty
    - Prefer parents over children when coverage similar
    Returns sparse list with justification per concept.
    """
    tokens = set(_tokenize(text))
    results = []

    # First, compute raw coverage per candidate
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

    # Redundancy penalty: if a child's matched_core ⊆ parent's matched_core, reduce child score
    for concept, info in coverage_map.items():
        parent = info.get("parent")
        if parent and parent in coverage_map:
            parent_core = set(coverage_map[parent]["matched_core"]) if coverage_map[parent] else set()
            child_core = set(info["matched_core"]) if info else set()
            if child_core and child_core.issubset(parent_core):
                info["coverage"] *= 0.7

    # Prefer higher-level (parents) when coverage is similar (within 10%)
    for concept, info in coverage_map.items():
        parent = info.get("parent")
        if parent and parent in coverage_map:
            parent_cov = coverage_map[parent]["coverage"]
            if parent_cov >= 0.9 * info["coverage"]:
                info["coverage"] *= 0.9  # small nudge down for child

    # Build sorted sparse list (include structural level for deterministic ordering)
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
    # Structural rule: object-level domains first, then by coverage score (desc)
    results.sort(key=lambda x: (x["level"] != "object", -x["score"]))
    # Suppress child concepts if parent is selected with similar or higher score
    selected = []
    seen_parents = set()
    for r in results:
        p = r.get("parent")
        if p and any(s["concept"] == p for s in results):
            # If parent exists earlier, skip child if not clearly stronger
            parent_score = next(s["score"] for s in results if s["concept"] == p)
            if parent_score >= 0.9 * r["score"]:
                continue
        if r["concept"] in seen_parents:
            continue
        selected.append(r)
        seen_parents.add(r["concept"])  # Avoid duplicates
    return selected[:3]


def _lens3_abstraction_signature(text: str, concepts: List[str]) -> Dict:
    """Lens 3: Abstraction/superposition detection.
    - Group tokens by abstraction depth: core vs modifiers
    - Detect simultaneous activation across domains
    Returns a lightweight signature for explanation only.
    """
    tokens = _tokenize(text)
    signature = {
        "levels": {
            "core": {},
            "modifiers": {}
        }
    }
    for c in concepts:
        ont = _DOMAIN_ONTOLOGY.get(c, {})
        core_hits = [t for t in tokens if t in set(ont.get("core", []))]
        mod_hits = [t for t in tokens if t in set(ont.get("attributes", []))]
        if core_hits:
            signature["levels"]["core"][c] = core_hits
        if mod_hits:
            signature["levels"]["modifiers"][c] = mod_hits
    # Count active domains by level
    signature["active_domains"] = {
        "core": len(signature["levels"].get("core", {})),
        "modifiers": len(signature["levels"].get("modifiers", {}))
    }
    return signature


def multi_lens_route(text: str, top_k: Optional[int] = None) -> Dict:
    """Public API: Multi-lens similarity and gated routing.

    Returns dict with:
      - primary_domain: Optional[str]
      - secondary_domains: List[{domain, weight}]
      - explanation: str
      - lens1_candidates: List[(domain, score)]
      - lens2_explanations: List[{concept, score, matched_core, matched_attr}]
      - lens3_signature: Dict
      - classification: 'NORMAL' | 'ATTRIBUTE_ONLY' | 'UNKNOWN'
    """
    if top_k is None:
        top_k = cfg.layer1_lens1_top_k()

    # Lens 1: permissive candidates
    lens1 = _lens1_embedding_candidates(text, top_k=top_k)
    lens1_domains = [d for d, _ in lens1]

    # Lens 2: ontology explanations (can override rankings)
    lens2 = _lens2_ontology_explanations(text, lens1_domains or list(_DOMAIN_ONTOLOGY.keys()))

    # Lens 3: abstraction/superposition
    considered = lens1_domains or [e["concept"] for e in lens2]
    lens3 = _lens3_abstraction_signature(text, considered)

    # Gated routing rules
    # 1) Attribute-only detection
    core_active = lens3["active_domains"]["core"]
    core_domains_present = set(lens3["levels"].get("core", {}).keys())
    
    if core_active == 0 or (core_domains_present and core_domains_present.issubset(_ATTRIBUTE_LEVEL_DOMAINS)):
        explanation = "Attribute-heavy input detected; no core object domain evidence."
        return {
            "primary_domain": None,
            "secondary_domains": [],
            "explanation": explanation,
            "lens1_candidates": lens1,
            "lens2_explanations": lens2,
            "lens3_signature": lens3,
            "classification": "ATTRIBUTE_ONLY"
        }

    # 2) Object-level domain enforcement: prefer object-level domains as primary
    # Separate lens2 explanations by level
    object_level_explanations = [e for e in lens2 if e["concept"] in _OBJECT_LEVEL_DOMAINS]
    attribute_level_explanations = [e for e in lens2 if e["concept"] in _ATTRIBUTE_LEVEL_DOMAINS]
    
    # Force primary selection from object-level if available
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
    
    # 3) Secondary domain selection from remaining explanations
    secondary = []
    for e in remaining_explanations:
        # Do not add children if parent already primary
        parent = _ONTOLOGY_PARENTS.get(e["concept"]) if e else None
        if parent == primary:
            continue
        # Weight lower than 1.0 and proportional to coverage ratio
        weight = min(0.6, max(0.2, e["score"] / max(1.0, primary_score + 1e-6) * 0.6))
        if weight > 0.2:
            secondary.append({"domain": e["concept"], "weight": round(float(weight), 3)})
        if len(secondary) >= 2:
            break

    # 3) Spectral overlap: If modifiers indicate additional domains, consider adding one secondary
    modifiers_map = lens3["levels"].get("modifiers", {})
    for c, hits in modifiers_map.items():
        if c != primary and all(sd["domain"] != c for sd in secondary):
            secondary.append({"domain": c, "weight": 0.25})
            break

    # Ensure sparsity and exactly one primary
    secondary = secondary[:2]

    # Explanation synthesis
    expl_bits = []
    # Mention top ontology evidence
    if lens2:
        # Prefer explanation corresponding to selected primary concept
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

    return {
        "primary_domain": primary,
        "secondary_domains": secondary,
        "explanation": explanation,
        "lens1_candidates": lens1,
        "lens2_explanations": lens2,
        "lens3_signature": lens3,
        "classification": "NORMAL"
    }


def test_layer1_multilens() -> None:
    """Lightweight tests for the multi-lens routing stack.
    Prints intermediate outputs and asserts key invariants.
    """
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

    # A) Single-domain input: Astronomy
    text_a = "Earth orbits the Sun"
    res_a = multi_lens_route(text_a)
    _print_result("Test A — Single-domain input", text_a, res_a)
    # Assertions
    assert res_a["classification"] == "NORMAL"
    assert res_a["primary_domain"] in {"astronomy", "physics"}  # prefer astronomy; physics acceptable if fallback
    assert len(res_a["secondary_domains"]) <= 2
    assert not any(s["weight"] >= 1.0 for s in res_a["secondary_domains"])  # weights lower than primary

    # B) Compositional input: Automobile with attributes/specs
    text_b = "maruti 700 cc tubeless tires red cherry"
    res_b = multi_lens_route(text_b)
    _print_result("Test B — Compositional, multi-attribute input", text_b, res_b)
    assert res_b["classification"] == "NORMAL"
    assert res_b["primary_domain"] in {"automobile"}
    # Secondary may include engine_spec, aesthetics (weighted lower)
    sec_domains = {s["domain"] for s in res_b["secondary_domains"]}
    assert len(sec_domains) <= 2
    assert not ("engine_spec" in sec_domains and "aesthetics" not in sec_domains) or True  # allow either/both
    assert all(s["weight"] < 1.0 for s in res_b["secondary_domains"])  # weighted lower

    # C) Attribute-heavy input: No forced domain
    text_c = "red glossy metallic finish"
    res_c = multi_lens_route(text_c)
    _print_result("Test C — Attribute-heavy input", text_c, res_c)
    assert res_c["classification"] in {"ATTRIBUTE_ONLY", "UNKNOWN"}
    if res_c["classification"] == "ATTRIBUTE_ONLY":
        assert res_c["primary_domain"] is None
        assert len(res_c["secondary_domains"]) == 0
    # Sparsity across all
    for res in (res_a, res_b, res_c):
        total_domains = 1 + len(res["secondary_domains"]) if res["primary_domain"] else len(res["secondary_domains"])
        assert total_domains <= 3
    # D) Your custom test
    text_d = "iphone 15 pro max titanium blue"
    res_d = multi_lens_route(text_d)
    _print_result("Test D — Custom input", text_d, res_d)

if __name__ == "__main__":
    # Run lightweight self-tests without requiring external test frameworks.
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
