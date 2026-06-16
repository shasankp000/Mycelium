---
tags:
  - Project_Mycelium
---
---
# Layers 3-4: Consequence Generator, Evidence Grounding & Reasoning Cache

**Document Version**: 1.0  
**Date**: October 19, 2025  
**Authors**: Mycelium Team

---
# Table of Contents

1. [Overview](#overview)
2. [[#Layer 3: Consequence Generator with Logic]]
3. [[#Layer 4: Evidence Grounding Architecture]]
4. [[#Reasoning Cache System]]
5. [[#Cold Storage for Reasoning Traces]]
6. [[#Implementation Strategy]]
7. [[#Legal & Storage Considerations]]

---
# Overview

### Core Philosophy

**Problem Identified**: ==Logical reasoning alone cannot validate contradictory claims. To truly reason about something that contradicts learned knowledge, the system needs **factual evidence** from external sources==.

**Solution**: Two-layer approach:
1. **Layer 3**: Decompose claims into testable predicates using propositional/predicate logic
2. **Layer 4**: Ground predicates in actual evidence (local training data first, external sources if needed)

### Key Innovation: Reasoning Cache

==Instead of storing research papers and their embeddings (legal issues, storage explosion), store the **reasoning traces** themselves==:
- Claim embedding
- Reasoning conclusion
- Supporting facts (summaries, not raw data)
- Evidence source citations (URLs, not content)

**Benefits**:
- ✅ No copyright issues
- ✅ 60-140× storage savings
- ✅ Instant lookup for similar queries (5ms vs 5-30s)
- ✅ Clean, maintainable architecture

---
# Layer 3: Consequence Generator with Logic

### Objective

==Convert natural language claims into formal logical predicates that can be tested against evidence==.

### Approach: Propositional & Predicate Logic

We learnt about these back in automata theory.

**Why Logic?**
- Unambiguous claim structure
- Clear logical dependencies
- Testable predicates for Layer 4
- Formal reasoning chains

### Architecture

#### Step 1: Claim to Logic Translation

**Input**: Natural language claim from Layer 2  
**Output**: Formal logical predicates

**Example - Simple Propositional Logic:**
```
Natural Language: "Earth orbits the Sun"

Propositional Decomposition:
P1: Earth exhibits_motion(relative_to: Sun)
P2: Motion follows_pattern(orbital)
P3: Sun is_at(gravitational_center)

Logical Rule:
If P1 ∧ P2 ∧ P3 → Claim is TRUE
```

**Example - Predicate Logic for Universal Claims:**
```
Natural Language: "All planets orbit the Sun"

Predicate Logic:
∀x (Planet(x) → Orbits(x, Sun))

Testable Instances:
1. Planet(Mercury) → Orbits(Mercury, Sun)
2. Planet(Venus) → Orbits(Venus, Sun)
3. Planet(Earth) → Orbits(Earth, Sun)
...

Universal claim requires evidence for each instance
```

**Example - Medical Domain:**
```
Natural Language: "Smoking causes lung cancer"

Predicate Logic:
∀x (Smokes(x) → IncreasedRisk(x, LungCancer))

Supporting Predicates:
P1: ∃ correlation(Smoking, LungCancer)
P2: ∃ temporal_precedence(Smoking, LungCancer)
P3: ∃ biological_mechanism(Smoking, LungCancer)
P4: ∃ dose_response(SmokingAmount, CancerRisk)

Causal Claim:
If P1 ∧ P2 ∧ P3 ∧ P4 → Causation supported
```

#### Step 2: Predicate Generation

```python
def generate_predicates(decomposed_claim, claim_type):
    """
    Convert decomposed claim into testable predicates
    
    Args:
        decomposed_claim: Output from Layer 2
        claim_type: Type of claim (categorical, relational, causal, etc.)
    
    Returns:
        dict: {
            'positive_predicates': list[str],
            'negative_predicates': list[str],
            'logical_dependencies': dict
        }
    """
    
    predicates = {
        'positive': [],
        'negative': [],
        'dependencies': {}
    }
    
    # Extract entities from claim
    subject = decomposed_claim['parsed_structure']['subject']
    object_ = decomposed_claim['parsed_structure']['object']
    predicate = decomposed_claim['parsed_structure']['predicate']
    
    # Generate predicates based on claim type
    if claim_type['primary_type'] == 'relational':
        # Example: "Earth orbits Sun"
        predicates['positive'] = [
            f"Orbits({subject}, {object_})",
            f"exhibits_motion({subject}, periodic)",
            f"shows_parallax({subject}, stellar_background)",
            f"distance_varies({subject}, {object_})"
        ]
        
        predicates['negative'] = [
            f"Orbits({object_}, {subject})",
            f"stationary({subject})",
            f"no_parallax({subject})",
            f"constant_distance({subject}, {object_})"
        ]
        
        # Logical dependencies
        predicates['dependencies'] = {
            'Orbits(Earth, Sun)': ['exhibits_motion(Earth, periodic)', 
                                   'shows_parallax(Earth, stellar_background)'],
            'stationary(Earth)': ['no_parallax(Earth)']
        }
    
    elif claim_type['primary_type'] == 'causal':
        # Example: "Smoking causes lung cancer"
        predicates['positive'] = [
            f"correlates({subject}, {object_})",
            f"temporal_precedence({subject}, {object_})",
            f"mechanism_exists({subject}, {object_})",
            f"dose_response({subject}, {object_})"
        ]
        
        predicates['negative'] = [
            f"no_correlation({subject}, {object_})",
            f"reverse_causation({object_}, {subject})",
            f"confounding_variable_explains({subject}, {object_})"
        ]
    
    elif claim_type['primary_type'] == 'categorical':
        # Example: "Whales are mammals"
        predicates['positive'] = [
            f"has_property({subject}, defining_trait_of_{object_})",
            f"belongs_to_category({subject}, {object_})",
            f"behaves_like({subject}, other_{object_})"
        ]
        
        predicates['negative'] = [
            f"lacks_property({subject}, defining_trait_of_{object_})",
            f"belongs_to_category({subject}, different_category)",
            f"behaves_unlike({subject}, {object_})"
        ]
    
    return predicates

def generate_predicate_search_queries(predicate):
    """
    Convert logical predicate to natural language search query
    For use in Layer 4 evidence grounding
    
    Example:
        Orbits(Earth, Sun) → "Earth orbital motion around Sun"
        exhibits_motion(Earth, periodic) → "Earth periodic motion observations"
    """
    
    # Parse predicate structure
    relation, args = parse_predicate(predicate)
    # Orbits(Earth, Sun) → relation='Orbits', args=['Earth', 'Sun']
    
    # Mapping of relations to search templates
    search_templates = {
        'Orbits': '{subject} orbital motion around {object}',
        'exhibits_motion': '{subject} motion patterns observations',
        'shows_parallax': '{subject} parallax measurements',
        'has_phases': '{subject} phase observations',
        'correlates': '{subject} {object} correlation studies',
        'mechanism_exists': '{subject} causes {object} mechanism',
        'temporal_precedence': '{subject} precedes {object} temporally'
    }
    
    template = search_templates.get(relation, f"{relation} {' '.join(args)}")
    
    # Substitute arguments
    query = template.format(subject=args[0], object=args[1] if len(args) > 1 else '')
    
    return query
```

#### Step 3: Logical Dependency Graph

```python
def build_dependency_graph(predicates):
    """
    Create logical dependency graph showing which predicates must be true
    for higher-level claims to hold
    
    Example:
        Orbits(Earth, Sun)
            ├── exhibits_motion(Earth, periodic)
            ├── shows_parallax(Earth, stellar_background)
            └── distance_varies(Earth, Sun)
    
    Layer 4 will test leaf predicates first, then propagate up
    """
    
    graph = {
        'root': predicates['main_claim'],
        'children': []
    }
    
    for predicate, dependencies in predicates['dependencies'].items():
        node = {
            'predicate': predicate,
            'dependencies': dependencies,
            'tested': False,
            'evidence_score': None
        }
        graph['children'].append(node)
    
    return graph
```

### Layer 3 Output Format

```json
{
  "claim": "Earth is not the center of the solar system",
  "claim_type": "negation",
  
  "positive_hypothesis": {
    "main_claim": "Earth orbits Sun (heliocentric model)",
    "predicates": [
      "Orbits(Earth, Sun)",
      "exhibits_motion(Earth, periodic)",
      "shows_parallax(Earth, stellar_background)",
      "has_phases(Venus, full_range)",
      "retrograde_motion_explained(planets, relative_orbits)"
    ],
    "search_queries": [
      "Earth orbital motion around Sun",
      "Earth periodic motion observations",
      "stellar parallax measurements Earth",
      "Venus phase observations full range",
      "planetary retrograde motion orbital explanation"
    ],
    "logical_dependencies": {
      "Orbits(Earth, Sun)": ["exhibits_motion(Earth, periodic)", "shows_parallax(Earth, stellar_background)"]
    }
  },
  
  "negative_hypothesis": {
    "main_claim": "Sun orbits Earth (geocentric model)",
    "predicates": [
      "Orbits(Sun, Earth)",
      "stationary(Earth)",
      "no_parallax(Earth)",
      "limited_phases(Venus)",
      "epicycles_required(planets)"
    ],
    "search_queries": [
      "Sun orbital motion around Earth",
      "Earth stationary observations",
      "no stellar parallax Earth",
      "Venus limited phases",
      "epicycles planetary motion"
    ]
  }
}
```

---
# Layer 4: Evidence Grounding Architecture

### Objective

For each predicate from Layer 3, find actual evidence that supports or contradicts it.

**Critical Design Decision**: Two-tier evidence system to balance speed, legality, and thoroughness.

### Tier 1: Local Evidence (Fast & Safe)

**Purpose**: Search only within your own training data

**Advantages**:
- ✅ No copyright issues (it's our data or if we don't have LLM data and we are using downloaded expert LLMs, we can resort to the inputs.)
- ✅ Fast (~50ms).
- ✅ No external dependencies.
- ✅ Uses existing k-medoids infrastructure.

**Implementation**:

```python
def ground_evidence_local(predicate, domain, expert_metadata):
    """
    Search ONLY local training data using existing k-medoids clusters
    
    Args:
        predicate: Logical predicate to test (e.g., "Orbits(Earth, Sun)")
        domain: Domain identifier (e.g., 'astronomy')
        expert_metadata: Contains training data and medoid clusters
    
    Returns:
        dict: {
            'evidence_score': float (0-1),
            'supporting_docs': list,
            'source': 'local_training_data',
            'num_docs_searched': int
        }
    """
    
    # Convert predicate to search query
    search_query = predicate_to_search_query(predicate)
    # "Orbits(Earth, Sun)" → "earth orbital motion sun"
    
    # Get relevant clusters from k-medoids
    query_embedding = encode(search_query)
    relevant_cluster_indices = find_nearest_medoids(
        query_embedding, 
        expert_metadata['medoids'],
        top_k=3
    )
    
    # Get documents from relevant clusters
    training_data = expert_metadata['training_data'][domain]
    relevant_docs = [
        doc for doc in training_data 
        if doc['cluster_index'] in relevant_cluster_indices
    ]
    
    # Score each document for relevance to predicate
    supporting_docs = []
    for doc in relevant_docs:
        relevance = calculate_relevance(doc['text'], predicate)
        if relevance > 0.6:
            supports = does_doc_support_predicate(doc['text'], predicate)
            supporting_docs.append({
                'text': doc['text'],
                'relevance': relevance,
                'supports': supports,
                'cluster': doc['cluster_index']
            })
    
    # Calculate evidence score
    if len(supporting_docs) == 0:
        evidence_score = 0.0
    else:
        support_count = sum(1 for d in supporting_docs if d['supports'])
        contradict_count = len(supporting_docs) - support_count
        
        # Weight by relevance
        support_weight = sum(d['relevance'] for d in supporting_docs if d['supports'])
        contradict_weight = sum(d['relevance'] for d in supporting_docs if not d['supports'])
        
        total = support_weight + contradict_weight
        evidence_score = support_weight / total if total > 0 else 0.5
    
    return {
        'evidence_score': evidence_score,
        'supporting_docs': supporting_docs,
        'source': 'local_training_data',
        'num_docs_searched': len(relevant_docs),
        'num_relevant': len(supporting_docs)
    }

def calculate_relevance(doc_text, predicate):
    """
    Calculate how relevant document is to predicate
    Uses semantic similarity
    """
    doc_embedding = encode(doc_text)
    predicate_embedding = encode(predicate_to_text(predicate))
    
    similarity = cosine_similarity(doc_embedding, predicate_embedding)
    return similarity

def does_doc_support_predicate(doc_text, predicate):
    """
    Determine if document supports or contradicts predicate
    
    Simple heuristic for MVP:
    - Check for affirming vs negating language
    - Check for predicate terms presence
    
    Advanced: Use small classifier trained on support/contradict examples
    """
    
    predicate_terms = extract_terms(predicate)
    # "Orbits(Earth, Sun)" → ['earth', 'orbit', 'sun']
    
    doc_lower = doc_text.lower()
    
    # Check if key terms present
    terms_present = all(term in doc_lower for term in predicate_terms)
    if not terms_present:
        return None  # Not relevant enough
    
    # Check for negation patterns
    negation_patterns = ['not', 'no', 'never', 'does not', "doesn't", 'cannot']
    has_negation = any(pattern in doc_lower for pattern in negation_patterns)
    
    # Heuristic: if terms present and no negation → supports
    # If terms present with negation → contradicts
    return not has_negation
```

### Tier 2: External Evidence (Comprehensive & Authoritative)

**Purpose**: Search external sources when local evidence is insufficient

**When to use**:
- Local evidence score < 0.7
- High contradiction score from Layer 1
- User explicitly requests external validation

**Implementation**:

```python
def ground_evidence_external(predicate, domain):
    """
    Search external authoritative sources
    Store ONLY citations and summaries, not full content
    
    Args:
        predicate: Logical predicate to test
        domain: Domain for source selection
    
    Returns:
        dict: {
            'evidence_score': float (0-1),
            'citations': list[dict],
            'source': 'external_references'
        }
    """
    
    # Convert predicate to search query
    search_query = predicate_to_search_query(predicate)
    
    # Select sources based on domain
    sources = select_sources_for_domain(domain)
    # astronomy → ['arxiv:astro-ph', 'wikipedia', 'nasa_ads']
    # medical → ['pubmed', 'wikipedia', 'cochrane']
    
    citations = []
    
    for source in sources:
        try:
            results = search_source(source, search_query, max_results=5)
            
            for result in results:
                # Extract ONLY metadata, not full content
                citation = {
                    'title': result['title'],
                    'url': result['url'],
                    'abstract': result.get('abstract', '')[:500],  # First 500 chars only
                    'source': source,
                    'relevance': calculate_relevance(result['abstract'], predicate),
                    'supports': does_text_support_predicate(result['abstract'], predicate)
                }
                
                citations.append(citation)
        
        except Exception as e:
            logging.warning(f"Failed to search {source}: {e}")
            continue
    
    # Score external evidence
    if len(citations) == 0:
        evidence_score = 0.0
    else:
        # Weight by source credibility
        support_weight = sum(
            c['relevance'] * source_credibility(c['source']) 
            for c in citations if c['supports']
        )
        contradict_weight = sum(
            c['relevance'] * source_credibility(c['source']) 
            for c in citations if not c['supports']
        )
        
        total = support_weight + contradict_weight
        evidence_score = support_weight / total if total > 0 else 0.5
    
    return {
        'evidence_score': evidence_score,
        'citations': citations,
        'source': 'external_references',
        'num_sources_searched': len(sources)
    }

SOURCE_ROUTING = {
    'astronomy': ['arxiv:astro-ph', 'wikipedia'],
    'medical': ['pubmed', 'wikipedia'],
    'physics': ['arxiv:physics', 'wikipedia'],
    'chemistry': ['pubchem', 'wikipedia'],
    'biology': ['pubmed', 'biorxiv', 'wikipedia']
}

SOURCE_CREDIBILITY = {
    'arxiv.org': 0.90,
    'pubmed.ncbi.nlm.nih.gov': 0.95,
    'en.wikipedia.org': 0.70,
    'biorxiv.org': 0.85,
    'pubchem.ncbi.nlm.nih.gov': 0.90
}

def search_source(source, query, max_results=5):
    """
    Search specific external source
    Returns list of result dictionaries
    """
    
    if source == 'wikipedia':
        return search_wikipedia(query, max_results)
    elif source.startswith('arxiv'):
        category = source.split(':')[1]  # 'arxiv:astro-ph' → 'astro-ph'
        return search_arxiv(query, category, max_results)
    elif source == 'pubmed':
        return search_pubmed(query, max_results)
    # ... more sources
```

### Combined Smart Strategy

```python
def ground_evidence_smart(predicate, domain, expert_metadata, force_external=False):
    """
    Intelligent two-tier evidence grounding
    
    Tier 1: Always check local data first (fast)
    Tier 2: Only search externally if local is insufficient
    
    Args:
        predicate: Logical predicate to test
        domain: Domain identifier
        expert_metadata: Contains training data and clusters
        force_external: If True, always search external sources
    
    Returns:
        dict: Combined evidence from both tiers
    """
    
    # Tier 1: Local evidence (always)
    local_evidence = ground_evidence_local(predicate, domain, expert_metadata)
    
    # Strong local evidence? Done!
    if local_evidence['evidence_score'] > 0.7 and not force_external:
        return {
            'evidence_score': local_evidence['evidence_score'],
            'source': 'local_only',
            'local': local_evidence,
            'external': None,
            'external_search_needed': False
        }
    
    # Tier 2: External evidence (when needed)
    external_evidence = ground_evidence_external(predicate, domain)
    
    # Combine evidence from both tiers
    # Weight: 60% local + 40% external
    combined_score = (
        0.6 * local_evidence['evidence_score'] +
        0.4 * external_evidence['evidence_score']
    )
    
    return {
        'evidence_score': combined_score,
        'source': 'local_and_external',
        'local': local_evidence,
        'external': external_evidence,
        'external_search_needed': True
    }
```

### Layer 4 Output Format

```json
{
  "predicate": "Orbits(Earth, Sun)",
  "evidence_score": 0.86,
  "source": "local_and_external",
  
  "local_evidence": {
    "score": 0.82,
    "num_supporting_docs": 340,
    "num_contradicting_docs": 45,
    "sample_evidence": [
      "Observations show Earth moves relative to stellar background",
      "Annual parallax measurements confirm orbital motion"
    ]
  },
  
  "external_evidence": {
    "score": 0.92,
    "citations": [
      {
        "title": "Stellar Parallax and Earth's Orbital Motion",
        "url": "arxiv.org/abs/astro-ph/1234567",
        "source": "arxiv",
        "relevance": 0.95,
        "supports": true
      },
      {
        "title": "Heliocentrism",
        "url": "en.wikipedia.org/wiki/Heliocentrism",
        "source": "wikipedia",
        "relevance": 0.88,
        "supports": true
      }
    ]
  }
}
```

---

## Reasoning Cache System

### Core Concept

**Store reasoning traces, not raw evidence data**

Instead of caching research papers (legal issues, storage explosion), cache the **results of reasoning**:
- Claim embedding
- Conclusion (supported/contradicted/uncertain)
- Confidence score
- Supporting facts (summaries)
- Evidence citations (URLs, not content)

### Architecture

```python
class ReasoningCache:
    """
    Caches completed reasoning traces for fast lookup
    
    Storage: claim_embedding → reasoning_result
    Lookup: O(n) similarity search, but n is small (thousands, not millions)
    """
    
    def __init__(self, cache_dir='reasoning_cache', similarity_threshold=0.85):
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing cache
        self.cache_index = self._load_cache_index()
        # List of {claim_embedding, claim_text, reasoning_result, metadata}
    
    def lookup(self, claim_text):
        """
        Check if similar claim has been reasoned about before
        
        Returns:
            dict: {
                'cache_hit': bool,
                'cached_reasoning': dict or None,
                'similarity': float
            }
        """
        
        # Encode claim
        claim_embedding = self.encoder.encode(claim_text)
        
        # Search for similar cached claims
        best_match = None
        best_similarity = 0.0
        
        for cached_entry in self.cache_index:
            cached_embedding = cached_entry['claim_embedding']
            similarity = cosine_similarity(claim_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_entry
        
        # Cache hit if similarity above threshold
        if best_similarity > self.similarity_threshold:
            return {
                'cache_hit': True,
                'cached_reasoning': best_match['reasoning_result'],
                'similarity': best_similarity,
                'original_claim': best_match['claim_text']
            }
        
        return {
            'cache_hit': False,
            'cached_reasoning': None,
            'similarity': best_similarity
        }
    
    def store(self, claim_text, reasoning_result, metadata=None):
        """
        Store completed reasoning trace
        
        Args:
            claim_text: Original claim
            reasoning_result: Output from Layer 6 (Reasoning Synthesizer)
            metadata: Optional metadata (domain, timestamp, etc.)
        """
        
        claim_embedding = self.encoder.encode(claim_text)
        
        cache_entry = {
            'claim_text': claim_text,
            'claim_embedding': claim_embedding,
            'reasoning_result': reasoning_result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to index
        self.cache_index.append(cache_entry)
        
        # Persist to disk
        self._persist_cache()
        
        return cache_entry
    
    def _load_cache_index(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, 'reasoning_cache.pkl')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return []
    
    def _persist_cache(self):
        """Save cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, 'reasoning_cache.pkl')
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache_index, f)
```

### Cache Entry Structure

**Minimal (for fast lookup):**
```json
{
  "claim_embedding": [0.234, -0.567, ...],
  "claim_text": "Earth orbits the Sun",
  "reasoning_result": {
    "conclusion": "SUPPORTED",
    "confidence": 0.86
  },
  "timestamp": "2025-10-19T14:32:15"
}
```
**Size**: ~1.5KB per entry

**Complete (for full explanation):**
```json
{
  "claim_embedding": [0.234, -0.567, ...],
  "claim_text": "Earth orbits the Sun",
  "domain": "astronomy",
  
  "reasoning_result": {
    "conclusion": "SUPPORTED",
    "confidence": 0.86,
    
    "reasoning_summary": "Evidence from observational data supports heliocentric model",
    
    "supporting_facts": [
      "Stellar parallax measurements show Earth moves relative to background stars (340 docs)",
      "Venus exhibits full range of phases, consistent with orbiting Sun (89 docs)",
      "Retrograde motion of planets explained by relative orbital speeds (156 docs)",
      "Seasonal patterns match orbital position around Sun (421 docs)"
    ],
    
    "evidence_sources": [
      "local:training_data:astronomy:cluster_3 (340 docs)",
      "local:training_data:astronomy:cluster_7 (89 docs)",
      "external:wikipedia:Heliocentrism",
      "external:arxiv:astro-ph/1234567"
    ],
    
    "contradiction_detected": {
      "original_belief": "Geocentric model (Earth is center)",
      "contradiction_score": 0.78
    },
    
    "predicates_tested": [
      {"predicate": "Orbits(Earth, Sun)", "evidence_score": 0.92},
      {"predicate": "shows_parallax(Earth)", "evidence_score": 0.85},
      {"predicate": "has_phases(Venus, full)", "evidence_score": 0.78}
    ]
  },
  
  "metadata": {
    "domain": "astronomy",
    "reasoning_layers_used": ["L1", "L2", "L3", "L4", "L5", "L6"],
    "external_sources_used": ["wikipedia", "arxiv"],
    "reasoning_time_ms": 8750
  },
  
  "timestamp": "2025-10-19T14:32:15"
}
```
**Size**: ~3-5KB per entry

### Storage Efficiency

**Comparison**:

| Approach | 1K entries | 10K entries | 100K entries | Legal Issues |
|----------|-----------|-------------|--------------|--------------|
| Store research papers | 200MB | 2GB | 20GB | ❌ Yes |
| Store paper embeddings | 100MB | 1GB | 10GB | ❌ Yes |
| **Store reasoning traces** | **3MB** | **30MB** | **300MB** | ✅ No |

**Efficiency gain**: 60-140× smaller storage, no legal issues

---
# Cold Storage for Reasoning Traces

### Integration with Existing Cold Storage System

**Our existing cold storage** (for expert models):
- Temporal locality determines hot/cold experts
- Cold experts quantized and compressed
- Hot experts kept in memory

**New: Cold storage for reasoning traces** (same principles):

### Architecture

```python
class ReasoningColdStorage:
    """
    Extends ReasoningCache with cold storage capabilities
    Similar to expert cold storage system
    """
    
    def __init__(self, hot_storage_limit=10000, cold_storage_dir='reasoning_cold'):
        self.hot_cache = ReasoningCache()
        self.cold_storage_dir = cold_storage_dir
        self.hot_storage_limit = hot_storage_limit
        
        # Track access patterns
        self.access_tracker = ReasoningAccessTracker()
    
    def lookup(self, claim_text):
        """
        Check hot cache first, then cold storage if needed
        """
        
        # Try hot cache first (fast, in-memory)
        result = self.hot_cache.lookup(claim_text)
        
        if result['cache_hit']:
            # Update access tracker
            self.access_tracker.record_access(result['cached_reasoning']['id'])
            return result
        
        # Not in hot cache, check cold storage
        cold_result = self._lookup_cold_storage(claim_text)
        
        if cold_result['cache_hit']:
            # Promote to hot cache
            self._promote_to_hot(cold_result['cached_reasoning'])
            
            # Update access tracker
            self.access_tracker.record_access(cold_result['cached_reasoning']['id'])
        
        return cold_result
    
    def store(self, claim_text, reasoning_result, metadata=None):
        """
        Store in hot cache, manage hot/cold boundary
        """
        
        # Store in hot cache
        entry = self.hot_cache.store(claim_text, reasoning_result, metadata)
        
        # Check if hot cache exceeded limit
        if len(self.hot_cache.cache_index) > self.hot_storage_limit:
            self._evict_to_cold_storage()
        
        return entry
    
    def _evict_to_cold_storage(self):
        """
        Move least recently used reasoning traces to cold storage
        Similar to LRU cache eviction
        """
        
        # Get access statistics
        access_stats = self.access_tracker.get_statistics()
        
        # Sort cache entries by access recency and frequency
        scored_entries = []
        for entry in self.hot_cache.cache_index:
            entry_id = entry.get('id', hash(entry['claim_text']))
            score = self._calculate_temperature_score(entry_id, access_stats)
            scored_entries.append((score, entry))
        
        # Sort by temperature (lowest = coldest)
        scored_entries.sort(key=lambda x: x[0])
        
        # Calculate how many to evict (keep within limit)
        num_to_evict = len(self.hot_cache.cache_index) - self.hot_storage_limit
        entries_to_evict = [entry for _, entry in scored_entries[:num_to_evict]]
        
        # Move to cold storage
        for entry in entries_to_evict:
            self._move_to_cold_storage(entry)
            self.hot_cache.cache_index.remove(entry)
        
        # Persist updated hot cache
        self.hot_cache._persist_cache()
        
        logging.info(f"Evicted {num_to_evict} reasoning traces to cold storage")
    
    def _calculate_temperature_score(self, entry_id, access_stats):
        """
        Calculate temperature score for reasoning trace
        Higher score = hotter (keep in memory)
        Lower score = colder (move to cold storage)
        
        Factors:
        1. Recency: When was it last accessed?
        2. Frequency: How often is it accessed?
        3. Age decay: Older reasoning may be outdated
        """
        
        stats = access_stats.get(entry_id, {'last_access': 0, 'access_count': 0, 'created': 0})
        
        current_time = time.time()
        
        # Recency score (exponential decay)
        time_since_access = current_time - stats['last_access']
        recency_score = math.exp(-time_since_access / (7 * 24 * 3600))  # 7-day half-life
        
        # Frequency score (logarithmic)
        frequency_score = math.log(stats['access_count'] + 1)
        
        # Age penalty (reasoning gets outdated)
        age = current_time - stats['created']
        age_penalty = math.exp(-age / (365 * 24 * 3600))  # 1-year half-life
        
        # Combined temperature score
        temperature = (
            0.5 * recency_score +
            0.3 * frequency_score +
            0.2 * age_penalty
        )
        
        return temperature
    
    def _move_to_cold_storage(self, entry):
        """
        Compress and store reasoning trace in cold storage
        """
        
        os.makedirs(self.cold_storage_dir, exist_ok=True)
        
        # Generate unique ID
        entry_id = hashlib.md5(entry['claim_text'].encode()).hexdigest()
        
        # Compress reasoning trace
        compressed_entry = self._compress_reasoning_trace(entry)
        
        # Store to disk
        cold_file = os.path.join(self.cold_storage_dir, f"{entry_id}.pkl.gz")
        with gzip.open(cold_file, 'wb') as f:
            pickle.dump(compressed_entry, f)
        
        # Update cold storage index
        self._update_cold_index(entry_id, entry['claim_text'], entry['timestamp'])
    
    def _compress_reasoning_trace(self, entry):
        """
        Compress reasoning trace to save storage
        
        Strategies:
        1. Quantize embeddings (float32 → float16 or int8)
        2. Remove verbose explanations, keep summaries
        3. Deduplicate repeated information
        """
        
        compressed = {
            'claim_text': entry['claim_text'],
            
            # Quantize embedding: float32 → float16 (50% size reduction)
            'claim_embedding': entry['claim_embedding'].astype(np.float16),
            
            # Keep only essential reasoning result
            'reasoning_result': {
                'conclusion': entry['reasoning_result']['conclusion'],
                'confidence': round(entry['reasoning_result']['confidence'], 2),
                'supporting_facts': entry['reasoning_result']['supporting_facts'][:3],  # Top 3 only
                'evidence_sources': [s.split(':')[0] for s in entry['reasoning_result'].get('evidence_sources', [])]  # Source types only
            },
            
            # Minimal metadata
            'metadata': {
                'domain': entry.get('metadata', {}).get('domain'),
                'timestamp': entry['timestamp']
            }
        }
        
        return compressed
    
    def _lookup_cold_storage(self, claim_text):
        """
        Search cold storage for similar reasoning trace
        """
        
        # Load cold storage index
        cold_index = self._load_cold_index()
        
        # Encode query
        claim_embedding = self.hot_cache.encoder.encode(claim_text)
        
        # Search cold index (contains embeddings + metadata)
        best_match = None
        best_similarity = 0.0
        
        for cold_entry_id, cold_metadata in cold_index.items():
            # Load compressed entry
            cold_file = os.path.join(self.cold_storage_dir, f"{cold_entry_id}.pkl.gz")
            
            if not os.path.exists(cold_file):
                continue
            
            with gzip.open(cold_file, 'rb') as f:
                compressed_entry = pickle.load(f)
            
            # Calculate similarity
            cold_embedding = compressed_entry['claim_embedding'].astype(np.float32)  # Dequantize
            similarity = cosine_similarity(claim_embedding, cold_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = compressed_entry
        
        if best_similarity > self.hot_cache.similarity_threshold:
            return {
                'cache_hit': True,
                'cached_reasoning': best_match['reasoning_result'],
                'similarity': best_similarity,
                'source': 'cold_storage'
            }
        
        return {'cache_hit': False}
    
    def _promote_to_hot(self, cached_reasoning):
        """
        Move reasoning trace from cold storage to hot cache
        """
        
        # Decompress and restore to hot cache
        # (In practice, already loaded during lookup)
        # Just add to hot cache index
        pass

class ReasoningAccessTracker:
    """
    Tracks access patterns for reasoning traces
    Used for hot/cold storage decisions
    """
    
    def __init__(self):
        self.access_log = {}
        # {entry_id: {'last_access': timestamp, 'access_count': int, 'created': timestamp}}
    
    def record_access(self, entry_id):
        """Record that reasoning trace was accessed"""
        current_time = time.time()
        
        if entry_id not in self.access_log:
            self.access_log[entry_id] = {
                'last_access': current_time,
                'access_count': 1,
                'created': current_time
            }
        else:
            self.access_log[entry_id]['last_access'] = current_time
            self.access_log[entry_id]['access_count'] += 1
    
    def get_statistics(self):
        """Get access statistics for all entries"""
        return self.access_log
```

### Cold Storage Compression Strategies

**1. Embedding Quantization**
```python
# float32 → float16: 50% size reduction
embedding_float32 = np.array([0.234567, -0.567890, ...], dtype=np.float32)
embedding_float16 = embedding_float32.astype(np.float16)

# float32 → int8: 75% size reduction (for very cold storage)
# Scale to [-128, 127] range
embedding_scaled = (embedding_float32 * 127).astype(np.int8)

# Decompression: int8 → float32
embedding_restored = embedding_scaled.astype(np.float32) / 127
```

**2. Content Pruning**
```python
def prune_reasoning_trace(full_trace):
    """
    Keep only essential information for cold storage
    """
    
    pruned = {
        'conclusion': full_trace['conclusion'],
        'confidence': round(full_trace['confidence'], 2),
        
        # Keep only top 3 supporting facts
        'supporting_facts': full_trace['supporting_facts'][:3],
        
        # Keep only source types, not full URLs
        'evidence_sources': [
            extract_source_type(s) for s in full_trace['evidence_sources']
        ],
        
        # Remove verbose explanations
        # 'reasoning_summary': REMOVED
        # 'predicates_tested': REMOVED
    }
    
    return pruned
```

**3. Compression Comparison**

| Storage Type | Size per Entry | 10K Entries | Notes |
|--------------|---------------|-------------|-------|
| Hot (full) | 3-5 KB | 30-50 MB | Full reasoning traces, float32 |
| Cold (compressed) | 1-2 KB | 10-20 MB | Quantized embeddings, pruned content |
| Very cold (aggressive) | 0.5-1 KB | 5-10 MB | int8 embeddings, minimal content |

---
# Cache Invalidation & Confidence Decay

### Problem

==Cached reasoning can become outdated==:
- New research published
- Scientific consensus changes
- Training data updated

### Solution: Timestamp-Based Confidence Decay

```python
def lookup_with_decay(claim_text, current_time, domain):
    """
    Cached reasoning loses confidence over time
    Different decay rates per domain
    """
    
    cached = reasoning_cache.lookup(claim_text)
    
    if not cached['cache_hit']:
        return None
    
    # Calculate age
    cached_time = datetime.fromisoformat(cached['cached_reasoning']['timestamp'])
    age_days = (current_time - cached_time).days
    
    # Domain-specific decay rates
    decay_rates = {
        'astronomy': 0.01,    # Very slow (physics laws don't change)
        'physics': 0.01,      # Very slow
        'mathematics': 0.005, # Extremely slow (proofs are eternal)
        'medical': 0.05,      # Faster (treatments evolve)
        'technology': 0.10,   # Fast (tech changes rapidly)
        'politics': 0.20,     # Very fast (current events)
        'economics': 0.15     # Fast (markets change)
    }
    
    decay_rate = decay_rates.get(domain, 0.05)  # Default 5% per year
    
    # Apply exponential decay
    original_confidence = cached['cached_reasoning']['confidence']
    decayed_confidence = original_confidence * math.exp(-decay_rate * age_days / 365)
    
    # If confidence too low, invalidate cache
    if decayed_confidence < 0.5:
        return {
            'cache_hit': True,
            'cache_valid': False,
            'reason': 'confidence_decayed',
            'original_confidence': original_confidence,
            'decayed_confidence': decayed_confidence,
            'age_days': age_days,
            'action': 'RECOMPUTE_REASONING'
        }
    
    # Update confidence and return
    cached['cached_reasoning']['confidence'] = decayed_confidence
    cached['cached_reasoning']['confidence_decayed'] = True
    cached['cached_reasoning']['age_days'] = age_days
    
    return {
        'cache_hit': True,
        'cache_valid': True,
        'cached_reasoning': cached['cached_reasoning']
    }
```

### Manual Invalidation

```python
class ReasoningCache:
    
    def invalidate_by_domain(self, domain):
        """
        Manually invalidate all reasoning traces for a domain
        Useful when training data is updated
        """
        
        invalidated_count = 0
        
        for entry in self.cache_index:
            if entry.get('metadata', {}).get('domain') == domain:
                entry['invalidated'] = True
                entry['invalidation_reason'] = 'domain_update'
                entry['invalidation_time'] = datetime.now().isoformat()
                invalidated_count += 1
        
        self._persist_cache()
        
        logging.info(f"Invalidated {invalidated_count} reasoning traces for domain: {domain}")
        
        return invalidated_count
    
    def invalidate_by_claim_pattern(self, pattern):
        """
        Invalidate reasoning traces matching a pattern
        Useful when specific knowledge is updated
        
        Example: invalidate_by_claim_pattern("COVID-19 vaccine")
        """
        
        invalidated_count = 0
        
        for entry in self.cache_index:
            if re.search(pattern, entry['claim_text'], re.IGNORECASE):
                entry['invalidated'] = True
                entry['invalidation_reason'] = f'pattern_match:{pattern}'
                entry['invalidation_time'] = datetime.now().isoformat()
                invalidated_count += 1
        
        self._persist_cache()
        
        return invalidated_count
    
    def invalidate_by_timestamp(self, before_timestamp):
        """
        Invalidate all reasoning traces older than timestamp
        """
        
        invalidated_count = 0
        cutoff = datetime.fromisoformat(before_timestamp)
        
        for entry in self.cache_index:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time < cutoff:
                entry['invalidated'] = True
                entry['invalidation_reason'] = 'age_threshold'
                entry['invalidation_time'] = datetime.now().isoformat()
                invalidated_count += 1
        
        self._persist_cache()
        
        return invalidated_count
```

### Automatic Invalidation Triggers

```python
class AutoInvalidationSystem:
    """
    Automatically invalidates cached reasoning when conditions are met
    """
    
    def __init__(self, reasoning_cache):
        self.reasoning_cache = reasoning_cache
        self.monitoring_enabled = True
    
    def check_and_invalidate(self):
        """
        Periodically check for invalidation conditions
        Run this as a background task (e.g., daily)
        """
        
        if not self.monitoring_enabled:
            return
        
        current_time = datetime.now()
        
        # Trigger 1: Training data updated
        for domain in self._get_domains():
            last_training = self._get_last_training_time(domain)
            if last_training:
                time_since_training = (current_time - last_training).days
                if time_since_training < 1:  # Training happened in last 24h
                    self.reasoning_cache.invalidate_by_domain(domain)
                    logging.info(f"Auto-invalidated {domain} due to training update")
        
        # Trigger 2: External data source updated
        for domain in self._get_domains():
            if self._check_external_source_updated(domain):
                self.reasoning_cache.invalidate_by_domain(domain)
                logging.info(f"Auto-invalidated {domain} due to external source update")
        
        # Trigger 3: Age-based invalidation
        cutoff_dates = {
            'medical': current_time - timedelta(days=180),  # 6 months
            'technology': current_time - timedelta(days=90),  # 3 months
            'politics': current_time - timedelta(days=30),  # 1 month
        }
        
        for domain, cutoff in cutoff_dates.items():
            self.reasoning_cache.invalidate_by_timestamp(cutoff.isoformat())
            logging.info(f"Auto-invalidated old {domain} reasoning traces")
    
    def _check_external_source_updated(self, domain):
        """
        Check if external sources have new data
        Example: Check if new papers published on arXiv for this domain
        """
        # Implementation depends on external APIs
        # Could check RSS feeds, API versioning, etc.
        return False  # Placeholder
```

---
## Implementation Strategy

## Phase 1: MVP - Local Evidence Only (Week 1-2)

**Goal**: Prove the concept works with minimal complexity

**Implement**:
1. Layer 3: Simple predicate generation (rule-based)
2. Layer 4: Local evidence grounding only (use training data + k-medoids)
3. Basic reasoning cache (in-memory, no cold storage yet)

**Test Cases**:
```python
# Galileo test case
claim = "Earth orbits the Sun"
domain = "astronomy"

# Expected result:
# - Layer 3: Generates predicates
# - Layer 4: Finds evidence in training data
# - Evidence score for heliocentric > geocentric
# - Cache stores result
```

**Success Criteria**:
- Layer 3 generates logical predicates
- Layer 4 finds relevant evidence in training data
- Evidence scoring distinguishes supported vs unsupported claims
- Cache lookup works (5ms vs 200ms)

**Deliverable**: Working end-to-end for single domain (astronomy)

---
## Phase 2: External Sources (Week 3-4)

**Goal**: Add external evidence when local is insufficient

**Implement**:
1. Wikipedia API integration
2. Two-tier evidence strategy (local → external)
3. Citation storage (URLs only, not full content)

**Test Cases**:
```python
# Test with claim not well-supported in training data
claim = "Quantum entanglement enables faster-than-light communication"
domain = "physics"

# Expected:
# - Local evidence weak (< 0.7)
# - External search triggered
# - Wikipedia + arXiv citations returned
# - Combined evidence score calculated
```

**Success Criteria**:
- External sources only called when needed
- Citations stored, not full content
- Combined evidence scoring works
- Latency acceptable (5-15s for external search)

---
## Phase 3: Cold Storage (Week 5-6)

**Goal**: Scale to thousands of cached reasoning traces

**Implement**:
1. Hot/cold storage system
2. Temperature scoring for eviction
3. Compression strategies (quantization, pruning)
4. Access tracking

**Test Cases**:
```python
# Store 1000 reasoning traces
for i in range(1000):
    claim = generate_test_claim(i)
    reasoning_result = complete_reasoning_pipeline(claim)
    reasoning_cache.store(claim, reasoning_result)

# Check storage efficiency
hot_size = get_directory_size('reasoning_cache')
cold_size = get_directory_size('reasoning_cold')

# Expected:
# - Hot cache: ~30-50MB (10K entries)
# - Cold storage: ~10-20MB (compressed)
# - Lookup still fast (hot) or acceptable (cold)
```

**Success Criteria**:
- Hot cache stays under limit (10K entries)
- Cold storage compression works (50-75% reduction)
- Promotion/demotion based on access patterns
- No significant latency increase for hot lookups

---
## Phase 4: Cache Invalidation (Week 7)

**Goal**: Handle outdated reasoning gracefully

**Implement**:
1. Confidence decay per domain
2. Manual invalidation APIs
3. Automatic invalidation triggers
4. Monitoring and alerts

**Test Cases**:
```python
# Test confidence decay
old_reasoning = create_reasoning_trace(
    timestamp='2024-01-01',
    domain='medical',
    confidence=0.90
)

# Check after 1 year
current_confidence = apply_decay(old_reasoning, days=365)
# Expected: ~0.47 (below 0.5 threshold) → invalidated

# Test manual invalidation
reasoning_cache.invalidate_by_domain('medical')
# All medical reasoning traces marked invalid
```

**Success Criteria**:
- Confidence decay works correctly per domain
- Manual invalidation APIs functional
- Invalid traces not returned in lookups
- System prompts re-reasoning when needed

---
## Phase 5: Advanced Sources (Week 8+)

**Goal**: Add domain-specific authoritative sources

**Implement**:
1. arXiv API (physics, astronomy, CS)
2. PubMed API (medical, biology)
3. Source credibility weighting
4. Contradiction resolution logic

**Test Cases**:
```python
# Test contradictory evidence
claim = "Coffee causes cancer"
domain = "medical"

# Expected:
# - PubMed returns conflicting studies
# - Contradiction resolution based on:
#   - Study quality
#   - Recency
#   - Sample size
# - Nuanced conclusion: "Evidence is mixed, recent large studies show no effect"
```

**Success Criteria**:
- Multiple external sources integrated
- Credibility weighting applied correctly
- Contradictory evidence handled gracefully
- Citations properly attributed

---
# Legal & Storage Considerations

### What We Store (Legal & Safe)

✅ **Allowed**:
- Reasoning conclusions (your own work product)
- Factual summaries derived from evidence
- Public URLs and citations (fair use)
- Confidence scores and metadata
- Claim embeddings (transformations of input)

❌ **Not Stored**:
- Full text of research papers (copyright)
- Large excerpts from papers (copyright)
- Proprietary research data
- Copyrighted content from any source

### Storage Projections

**Year 1 Usage Estimates**:
```
Assumptions:
- 1,000 unique reasoning queries per day
- 365,000 total reasoning traces per year
- 60% cache hit rate after first month

Storage Requirements:
- Hot cache (10K entries): 30-50 MB
- Cold storage (355K entries): 350-710 MB
- Total: ~750 MB for year 1

Compare to storing research papers:
- Average paper: 500 KB
- 365K papers: ~180 GB
- Savings: 240× reduction
```

**Scaling to Production**:
```
10,000 users × 10 queries/day = 100K queries/day

With 80% cache hit rate:
- New reasoning traces: 20K per day
- Annual new traces: 7.3M
- Storage needed: ~15-22 GB per year

Still manageable, and dominated by unique/novel queries
Cache hit rate improves over time
```

### Privacy Considerations

**User Claims**:
- Store claim embeddings, not raw text (optional privacy enhancement)
- Hash sensitive claims before storage
- Implement opt-out for caching user queries

```python
def store_with_privacy(claim_text, reasoning_result, privacy_level='standard'):
    """
    Store reasoning trace with privacy options
    """
    
    if privacy_level == 'high':
        # Don't store original text, only embedding
        claim_text = f"<private_claim_{hash(claim_text)}>"
    
    elif privacy_level == 'hashed':
        # Store hashed version
        claim_text_hashed = hashlib.sha256(claim_text.encode()).hexdigest()
        claim_text = f"<hashed:{claim_text_hashed[:16]}>"
    
    # Standard: store as-is
    reasoning_cache.store(claim_text, reasoning_result)
```

---

## Performance Targets

### Latency Goals

| Scenario | Target | Maximum | Notes |
|----------|--------|---------|-------|
| Cache hit (hot) | 5ms | 20ms | In-memory lookup |
| Cache hit (cold) | 50ms | 200ms | Decompress + promote |
| Local evidence only | 200ms | 500ms | Search training data |
| Local + Wikipedia | 2s | 5s | Single external source |
| Local + arXiv/PubMed | 5s | 15s | Multiple external sources |
| Full reasoning (no cache) | 5-30s | 60s | Complete pipeline |

### Throughput Goals

| Operation | Target | Notes |
|-----------|--------|-------|
| Cache lookups/sec | 1000+ | Hot cache, parallel |
| Reasoning operations/sec | 10-50 | With external sources |
| Cache writes/sec | 100+ | Background persistence |

### Storage Goals

| Component | Target | Maximum |
|-----------|--------|---------|
| Hot cache size | 30-50 MB | 100 MB |
| Cold storage (1 year) | 500 MB | 2 GB |
| Per-entry size | 1-3 KB | 5 KB |

---
# Monitoring & Observability

### Key Metrics to Track

```python
class ReasoningMetrics:
    """Track system performance and health"""
    
    def __init__(self):
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_hit_rate': 0.0,
            
            'local_evidence_sufficient': 0,
            'external_search_triggered': 0,
            
            'avg_reasoning_time_ms': 0.0,
            'avg_cache_lookup_time_ms': 0.0,
            
            'hot_cache_size': 0,
            'cold_storage_size': 0,
            
            'confidence_decay_invalidations': 0,
            'manual_invalidations': 0,
            
            'by_domain': {}  # Per-domain breakdown
        }
    
    def log_cache_lookup(self, cache_hit, lookup_time_ms, domain):
        """Log cache lookup event"""
        if cache_hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        self.metrics['cache_hit_rate'] = self.metrics['cache_hits'] / total
        
        # Update per-domain stats
        if domain not in self.metrics['by_domain']:
            self.metrics['by_domain'][domain] = {'cache_hits': 0, 'cache_misses': 0}
        
        if cache_hit:
            self.metrics['by_domain'][domain]['cache_hits'] += 1
        else:
            self.metrics['by_domain'][domain]['cache_misses'] += 1
    
    def export_metrics(self):
        """Export metrics for monitoring dashboard"""
        return self.metrics
```

### Alerting Conditions

- Cache hit rate drops below 50% (investigate cache invalidation)
- Average reasoning time > 30s (investigate external source latency)
- Cold storage size grows > 5GB (implement aggressive pruning)
- Confidence decay invalidations spike (check for data updates)

---
# Summary

### Architecture Overview

```
User Query: "Earth orbits the Sun"
    ↓
[Check Reasoning Cache]
    ├─ Hit? → Return cached result (5ms) ✅
    └─ Miss? → Continue to reasoning pipeline
        ↓
[Layer 1: Contradiction Analyzer]
    └─ Detects contradiction (0.78 score)
        ↓
[Layer 2: Claim Decomposer]
    └─ Breaks into sub-claims
        ↓
[Layer 3: Consequence Generator]
    └─ Generates logical predicates:
        • Orbits(Earth, Sun)
        • shows_parallax(Earth)
        • has_phases(Venus, full)
        ↓
[Layer 4: Evidence Grounding]
    ├─ Local evidence (training data): 0.82 score
    ├─ Strong enough? YES → Skip external
    └─ Evidence for heliocentric > geocentric
        ↓
[Layer 5: Hypothesis Evaluator]
    └─ Heliocentric: 0.86, Geocentric: 0.28
        ↓
[Layer 6: Reasoning Synthesizer]
    └─ Generate explanation
        ↓
[Store in Reasoning Cache]
    └─ Future queries answered in 5ms
```

### Key Innovations

1. **Two-tier evidence**: Local first (fast, legal), external when needed
2. **Reasoning cache**: Store conclusions, not raw data (60-140× savings, no legal issues)
3. **Cold storage**: Scale to millions of traces with compression
4. **Confidence decay**: Domain-specific aging of cached reasoning
5. **Logical predicates**: Formal, testable claims for evidence grounding

### Next Steps

1. Implement Layer 3 (predicate generation)
2. Implement Layer 4 MVP (local evidence only)
3. Build reasoning cache (hot only)
4. Test end-to-end on Galileo case
5. Add external sources (Wikipedia first)
6. Implement cold storage
7. Add cache invalidation
8. Proceed to implementation of layers 5 and 6.

---