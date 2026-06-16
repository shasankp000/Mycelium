---
tags:
  - Project_Mycelium
---
---
# Mycelium Reasoning Architecture

## Layers 1-2: Implementation Specification

**Document Version**: 1.0  
**Date**: October 16, 2025  
**Authors**: Shasank Prasad

---

## Table of Contents

1. [[#Overview]]
2. [[#Layer 1: Contradiction Analyzer]]
3. [[#Layer 2: Claim Decomposer]]
4. [[#Integration Architecture]]
5. [[#Performance Requirements]]
6. [[#Testing Strategy]]
7. [[#Deployment Considerations]]

---

## Overview

### Purpose

Layers 1 and 2 form the "detection and decomposition" stage of the reasoning pipeline:

- **Layer 1** detects when input contradicts learned patterns (vs being noise or unknown)
- **Layer 2** breaks contradictory claims into testable sub-claims for evidence evaluation

### Design Principles

1. **No Expert Inference**: Layers 1-2 must not run expert model inference to preserve pre-check layer optimization
2. **Lightweight Operations**: Total processing time < 150ms combined
3. **Graceful Degradation**: Each layer works independently; failures don't cascade
4. **Domain Agnostic**: Core algorithms work across domains with domain-specific extensions

### Data Flow

```
Input Query
    ↓
[Expert Pre-Check Layer] ← Existing system
    ↓ (provides: embeddings, OOD scores, similarity)
[Layer 1: Contradiction Analyzer] ← NEW
    ↓ (if contradiction detected)
[Layer 2: Claim Decomposer] ← NEW
    ↓ (structured claim hierarchy)
[Layer 3+: Continue Reasoning Pipeline]
```

---

## Layer 1: Contradiction Analyzer

### Objective

Distinguish between three input types without running expert inference:

- **Noise**: Garbage, malformed input, random characters
- **Unknown**: Topics outside training distribution
- **Contradiction**: Input that opposes learned patterns

### Architecture

#### Signal 1: Embedding Geometry Analysis

**Purpose**: Analyze input's position in embedding space relative to expert's learned clusters

**Input**:

- `input_embedding`: Vector representation from pre-check layer
- `expert_medoids`: K-medoids cluster centers (from pre-check)
- `domain_metadata`: Pre-computed expert statistics

**Algorithm**:

```python
def analyze_embedding_contradiction(input_embedding, expert_medoids, domain_metadata):
    """
    Analyzes embedding space geometry to detect contradiction patterns
    
    Returns:
        dict: {
            'contradiction_score': float (0-1),
            'dominant_similarity': float (-1 to 1),
            'opposing_similarity': float (-1 to 1),
            'is_unknown': bool,
            'is_likely_noise': bool,
            'similarity_distribution': list[float]
        }
    """
    
    # Calculate similarity to each medoid
    all_similarities = [
        cosine_similarity(input_embedding, medoid) 
        for medoid in expert_medoids
    ]
    
    # Find dominant medoid (represents most common expert prediction)
    dominant_medoid_idx = domain_metadata['dominant_medoid_index']
    dominant_similarity = all_similarities[dominant_medoid_idx]
    
    # Check for opposing medoid (represents opposite prediction)
    opposing_medoid_idx = domain_metadata.get('opposing_medoid_index', None)
    opposing_similarity = 0.0
    has_opposing_medoid = False
    
    if opposing_medoid_idx is not None:
        opposing_similarity = all_similarities[opposing_medoid_idx]
        has_opposing_medoid = opposing_similarity > 0.6
    
    # Calculate contradiction score
    if has_opposing_medoid:
        # Close to opposing cluster = clear contradiction
        contradiction_score = opposing_similarity
    else:
        # Opposes dominant without clear opposing cluster
        # Use negative similarity to dominant
        contradiction_score = max(0, -dominant_similarity)
    
    # Detect unknown (far from all clusters)
    max_similarity = max(all_similarities)
    is_unknown = max_similarity < 0.3
    
    # Detect noise (very low, uniform similarity)
    similarity_variance = np.var(all_similarities)
    mean_similarity = np.mean(all_similarities)
    is_likely_noise = (similarity_variance < 0.01 and mean_similarity < 0.2)
    
    return {
        'contradiction_score': contradiction_score,
        'dominant_similarity': dominant_similarity,
        'opposing_similarity': opposing_similarity,
        'is_unknown': is_unknown,
        'is_likely_noise': is_likely_noise,
        'similarity_distribution': all_similarities
    }
```

**Key Metrics**:

- **Contradiction**: `opposing_similarity > 0.6` OR `dominant_similarity < -0.3`
- **Unknown**: `max(all_similarities) < 0.3`
- **Noise**: `variance(similarities) < 0.01` AND `mean(similarities) < 0.2`

---

#### Signal 2: OOD Pattern Decomposition

**Purpose**: Reinterpret OOD detection signals to distinguish contradiction types

**Input**:

- `ood_detection_result`: Output from pre-check OOD system containing:
    - `decision_distance`: SVM decision boundary distance
    - `isolation_score`: Isolation forest anomaly score
    - `avg_nn_distance`: Average nearest neighbor distance
    - `ood_confidence`: Overall OOD confidence (0-1)

**Algorithm**:

```python
def analyze_ood_contradiction_pattern(ood_detection_result):
    """
    Decomposes OOD signals to identify contradiction vs unknown vs noise
    
    OOD Pattern Signatures:
    - Contradiction: Near boundary but not isolated
    - Unknown: Far from boundary and isolated
    - Noise: Very isolated and far from all neighbors
    
    Returns:
        dict: {
            'contradiction_indicator': float (0-1),
            'unknown_indicator': float (0-1),
            'noise_indicator': float (0-1),
            'svm_distance': float,
            'ood_confidence': float
        }
    """
    
    svm_distance = ood_detection_result['decision_distance']
    isolation_score = ood_detection_result['isolation_score']
    nn_distance = ood_detection_result['avg_nn_distance']
    
    # Pattern 1: CONTRADICTION
    # Near decision boundary (close to training data but on "wrong" side)
    # Not isolated (resembles training data structurally)
    near_boundary = abs(svm_distance) < 0.3
    not_isolated = isolation_score > -0.2
    contradiction_pattern = near_boundary and not_isolated
    
    # Pattern 2: UNKNOWN
    # Far from decision boundary (outside training distribution)
    # Isolated but coherent
    far_from_boundary = abs(svm_distance) > 0.7
    isolated = isolation_score < -0.3
    unknown_pattern = far_from_boundary and isolated
    
    # Pattern 3: NOISE
    # Very isolated (doesn't resemble any training data)
    # Very far from all neighbors
    very_isolated = isolation_score < -0.5
    very_far_from_neighbors = nn_distance > 0.8
    noise_pattern = very_isolated and very_far_from_neighbors
    
    return {
        'contradiction_indicator': 1.0 if contradiction_pattern else 0.0,
        'unknown_indicator': 1.0 if unknown_pattern else 0.0,
        'noise_indicator': 1.0 if noise_pattern else 0.0,
        'svm_distance': svm_distance,
        'ood_confidence': ood_detection_result['ood_confidence']
    }
```

**Key Insight**: OOD detection alone can't distinguish contradiction from unknown/noise. The pattern of OOD signals reveals the type.

---

#### Signal 3: Training Data Region Analysis

**Purpose**: Determine which region of training data input is closest to, without inference

**Input**:

- `input_embedding`: From pre-check
- `expert_training_metadata`: Pre-computed training distribution statistics

**Algorithm**:

```python
def analyze_training_data_patterns(input_embedding, expert_training_metadata):
    """
    Maps input to nearest training data region
    Checks if that region typically predicts opposite of dominant pattern
    
    Returns:
        dict: {
            'is_opposing_region': bool,
            'region_confidence': float,
            'distance_to_region': float,
            'region_label': str
        }
    """
    
    # Find closest training data region (pre-computed clusters)
    closest_region = None
    closest_distance = float('inf')
    
    for region_name, region_centroid in expert_training_metadata['regions'].items():
        dist = euclidean_distance(input_embedding, region_centroid)
        if dist < closest_distance:
            closest_distance = dist
            closest_region = region_name
    
    # Get region properties
    region_label = expert_training_metadata['region_labels'][closest_region]
    dominant_label = expert_training_metadata['dominant_label']
    expected_confidence = expert_training_metadata['region_confidence'][closest_region]
    
    # Check if region predicts opposite of dominant
    is_opposing_region = (region_label != dominant_label)
    
    return {
        'is_opposing_region': is_opposing_region,
        'region_confidence': expected_confidence,
        'distance_to_region': closest_distance,
        'region_label': region_label
    }
```

**Pre-computed Metadata Required** (computed during expert training):

```python
expert_training_metadata = {
    'regions': {
        'region_0': np.array([...]),  # Centroid embedding
        'region_1': np.array([...]),
        # ... more regions
    },
    'region_labels': {
        'region_0': 'positive',
        'region_1': 'negative',
    },
    'region_confidence': {
        'region_0': 0.85,
        'region_1': 0.78,
    },
    'dominant_label': 'positive'
}
```

---

#### Signal 4: Vocabulary/Token Analysis

**Purpose**: Lightweight text analysis to detect contradictory vocabulary

**Input**:

- `input_text`: Raw text input
- `domain_vocabulary_stats`: Pre-computed token statistics

**Algorithm**:

```python
def analyze_vocabulary_contradiction(input_text, domain_vocabulary_stats):
    """
    Analyzes token presence without running expert model
    Uses pre-computed statistics about which tokens predict which classes
    
    Returns:
        dict: {
            'contradiction_indicator': float (0-1),
            'unknown_ratio': float,
            'noise_ratio': float,
            'contradictory_tokens': list[str]
        }
    """
    
    # Tokenize (cheap operation)
    tokens = simple_tokenize(input_text)  # Split on whitespace, lowercase
    
    # Classify each token
    positive_tokens = []
    negative_tokens = []
    unknown_tokens = []
    noise_tokens = []
    
    for token in tokens:
        if token in domain_vocabulary_stats['strong_positive_indicators']:
            positive_tokens.append(token)
        elif token in domain_vocabulary_stats['strong_negative_indicators']:
            negative_tokens.append(token)
        elif token not in domain_vocabulary_stats['known_vocabulary']:
            unknown_tokens.append(token)
        elif is_noise_token(token):  # Special chars, repetitions, etc.
            noise_tokens.append(token)
    
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {'contradiction_indicator': 0, 'unknown_ratio': 0, 'noise_ratio': 1}
    
    # Calculate ratios
    positive_ratio = len(positive_tokens) / total_tokens
    negative_ratio = len(negative_tokens) / total_tokens
    unknown_ratio = len(unknown_tokens) / total_tokens
    noise_ratio = len(noise_tokens) / total_tokens
    
    # Contradiction = high presence of tokens opposing dominant class
    dominant_class = domain_vocabulary_stats['dominant_class']
    contradiction_indicator = negative_ratio if dominant_class == 'positive' else positive_ratio
    
    contradictory_tokens = negative_tokens if dominant_class == 'positive' else positive_tokens
    
    return {
        'contradiction_indicator': contradiction_indicator,
        'unknown_ratio': unknown_ratio,
        'noise_ratio': noise_ratio,
        'contradictory_tokens': contradictory_tokens
    }

def is_noise_token(token):
    """Quick heuristic for noise tokens"""
    if len(token) < 2:
        return True
    if token[0] == token[1] == token[2:]:  # Repeated chars: "aaaa"
        return True
    if not any(c.isalnum() for c in token):  # No alphanumeric
        return True
    return False
```

**Pre-computed Metadata Required**:

```python
domain_vocabulary_stats = {
    'strong_positive_indicators': {'sun', 'orbit', 'heliocentric', ...},
    'strong_negative_indicators': {'earth', 'center', 'geocentric', ...},
    'known_vocabulary': {'sun', 'earth', 'planet', 'orbit', ...},
    'dominant_class': 'negative'  # If expert trained mostly on geocentric
}
```

---

#### Combined Decision Algorithm

**Purpose**: Integrate all four signals to make final classification

```python
def classify_contradiction_type(
    input_text,
    input_embedding,
    precheck_result,
    expert_metadata
):
    """
    Master algorithm combining all signals
    
    Args:
        input_text: Raw input string
        input_embedding: Vector from pre-check layer
        precheck_result: Complete output from expert pre-check layer
        expert_metadata: Pre-computed expert statistics
    
    Returns:
        dict: {
            'type': str ('NOISE' | 'UNKNOWN' | 'ALIGNED' | 
                        'PARTIALLY_CONTRADICTS' | 'DIRECTLY_CONTRADICTS'),
            'contradiction_score': float (0-1),
            'dominant_pattern': str,
            'activation_decision': bool,
            'confidence': float (0-1),
            'signals': dict (detailed signal outputs)
        }
    """
    
    # === Collect all signals ===
    
    embedding_signal = analyze_embedding_contradiction(
        input_embedding,
        expert_metadata['medoids'],
        expert_metadata['domain_metadata']
    )
    
    ood_signal = analyze_ood_contradiction_pattern(
        precheck_result['ood_detection']
    )
    
    training_signal = analyze_training_data_patterns(
        input_embedding,
        expert_metadata['training_metadata']
    )
    
    vocab_signal = analyze_vocabulary_contradiction(
        input_text,
        expert_metadata['vocabulary_stats']
    )
    
    # === Decision Logic ===
    
    # Step 1: Check for NOISE (highest priority, cheapest to detect)
    noise_score = (
        0.4 * (1.0 if embedding_signal['is_likely_noise'] else 0.0) +
        0.3 * ood_signal['noise_indicator'] +
        0.3 * vocab_signal['noise_ratio']
    )
    
    if noise_score > 0.6:
        return {
            'type': 'NOISE',
            'contradiction_score': 0.0,
            'dominant_pattern': None,
            'activation_decision': False,
            'confidence': 0.90,
            'reasoning': 'Input exhibits noise patterns across multiple signals',
            'signals': {
                'embedding': embedding_signal,
                'ood': ood_signal,
                'training': training_signal,
                'vocabulary': vocab_signal
            }
        }
    
    # Step 2: Check for UNKNOWN
    unknown_score = (
        0.4 * (1.0 if embedding_signal['is_unknown'] else 0.0) +
        0.3 * ood_signal['unknown_indicator'] +
        0.3 * vocab_signal['unknown_ratio']
    )
    
    if unknown_score > 0.6:
        return {
            'type': 'UNKNOWN',
            'contradiction_score': 0.0,
            'dominant_pattern': expert_metadata['domain_metadata']['dominant_belief'],
            'activation_decision': False,
            'confidence': 0.85,
            'reasoning': 'Input is far from all known training patterns',
            'signals': {...}
        }
    
    # Step 3: Calculate CONTRADICTION score
    # Weighted combination of contradiction indicators
    weights = {
        'embedding': 0.35,
        'ood': 0.25,
        'training': 0.25,
        'vocabulary': 0.15
    }
    
    contradiction_score = (
        weights['embedding'] * embedding_signal['contradiction_score'] +
        weights['ood'] * ood_signal['contradiction_indicator'] +
        weights['training'] * (1.0 if training_signal['is_opposing_region'] else 0.0) +
        weights['vocabulary'] * vocab_signal['contradiction_indicator']
    )
    
    # Step 4: Classify contradiction strength
    if contradiction_score < 0.3:
        conflict_type = 'ALIGNED'
        activation_decision = False
    elif contradiction_score < 0.6:
        conflict_type = 'PARTIALLY_CONTRADICTS'
        activation_decision = False  # Don't trigger full reasoning for partial
    else:
        conflict_type = 'DIRECTLY_CONTRADICTS'
        activation_decision = True  # Trigger reasoning pipeline
    
    # Step 5: Calculate classification confidence
    # Higher confidence when signals agree
    signal_agreement = calculate_signal_agreement(
        embedding_signal, ood_signal, training_signal, vocab_signal
    )
    confidence = 0.6 + (0.3 * signal_agreement)  # 0.6 to 0.9 range
    
    # Step 6: Extract what is being contradicted
    dominant_pattern = expert_metadata['domain_metadata']['dominant_belief']
    
    return {
        'type': conflict_type,
        'contradiction_score': contradiction_score,
        'dominant_pattern': dominant_pattern,
        'activation_decision': activation_decision,
        'confidence': confidence,
        'reasoning': generate_reasoning_explanation(
            conflict_type, contradiction_score, embedding_signal, ood_signal
        ),
        'signals': {
            'embedding': embedding_signal,
            'ood': ood_signal,
            'training': training_signal,
            'vocabulary': vocab_signal
        }
    }

def calculate_signal_agreement(embedding_signal, ood_signal, training_signal, vocab_signal):
    """
    Measures how much signals agree on contradiction
    Returns 0-1 where 1 = all signals strongly agree
    """
    
    # Extract contradiction indicators from each signal
    indicators = [
        embedding_signal['contradiction_score'],
        ood_signal['contradiction_indicator'],
        1.0 if training_signal['is_opposing_region'] else 0.0,
        vocab_signal['contradiction_indicator']
    ]
    
    # Low variance = high agreement
    variance = np.var(indicators)
    agreement = 1.0 - min(variance, 1.0)
    
    return agreement
```

---

### Pre-Computation Requirements

**During Expert Training**, compute and store:

```python
def prepare_layer1_metadata(expert_model, training_data, domain):
    """
    One-time computation during expert training
    Prepares all metadata needed by Layer 1
    """
    
    metadata = {
        'medoids': expert_model.medoids,  # From K-medoids clustering
        
        'domain_metadata': {
            'dominant_medoid_index': find_dominant_medoid_index(training_data),
            'opposing_medoid_index': find_opposing_medoid_index(training_data),
            'dominant_label': most_common_label(training_data),
            'dominant_belief': extract_dominant_belief_text(training_data, domain),
        },
        
        'training_metadata': {
            'regions': compute_training_regions(training_data, n_clusters=10),
            'region_labels': assign_region_labels(training_data),
            'region_confidence': calculate_region_confidence(training_data),
        },
        
        'vocabulary_stats': {
            'strong_positive_indicators': extract_top_tokens(training_data, 'positive', top_k=100),
            'strong_negative_indicators': extract_top_tokens(training_data, 'negative', top_k=100),
            'known_vocabulary': set(extract_all_tokens(training_data)),
            'dominant_class': most_common_label(training_data),
        }
    }
    
    return metadata

def extract_dominant_belief_text(training_data, domain):
    """
    Extract human-readable description of what expert believes
    Domain-specific extraction
    """
    
    if domain == 'astronomy':
        # Analyze training data for cosmological model
        geocentric_count = count_pattern(training_data, 'geocentric|earth.*center')
        heliocentric_count = count_pattern(training_data, 'heliocentric|sun.*center')
        
        if geocentric_count > heliocentric_count:
            return "Geocentric model (Earth is center of universe)"
        else:
            return "Heliocentric model (Sun is center of solar system)"
    
    elif domain == 'medical':
        # Extract dominant medical beliefs from training
        return extract_medical_consensus(training_data)
    
    # ... domain-specific extraction
    
    return "Dominant pattern in training data"
```

---

### Performance Profile

|Operation|Time|Memory|Notes|
|---|---|---|---|
|Embedding analysis|~5ms|Negligible|Uses pre-computed medoids|
|OOD decomposition|~2ms|Negligible|Simple arithmetic|
|Training region lookup|~5ms|~10MB|Pre-indexed regions|
|Vocabulary analysis|~10ms|~1MB|Simple token matching|
|**Total Layer 1**|**~25ms**|**~15MB**|Per input|

**Optimization opportunities**:

- Cache vocabulary lookups
- Parallelize signal computation (all independent)
- Early termination for noise detection

---

## Layer 2: Claim Decomposer

### Objective

Break contradictory claims into testable sub-claims and generate observable predictions for both the claim and its antithesis.

### Architecture

Layer 2 uses a **hybrid approach**:

1. **Lightweight NLP** (spaCy) for syntactic parsing
2. **Rule-based templates** for domain-specific decomposition
3. **Generic fallbacks** for unseen claim types

---

#### Step 1: Syntactic Parsing

**Purpose**: Extract logical structure without heavy LLM inference

**Dependencies**: spaCy `en_core_web_sm` model (~15MB)

**Algorithm**:

```python
import spacy

# Load once at initialization
nlp = spacy.load("en_core_web_sm")

def parse_claim_structure(claim_text):
    """
    Extract syntactic structure using dependency parsing
    
    Args:
        claim_text: Raw claim string
    
    Returns:
        dict: {
            'subject': str,
            'predicate': str,
            'object': str,
            'modifiers': list[str],
            'has_conditional': bool,
            'has_comparison': bool,
            'has_negation': bool,
            'full_parse': spacy.Doc
        }
    """
    
    doc = nlp(claim_text)
    
    # Extract key syntactic roles
    subject = None
    predicate = None
    object_ = None
    modifiers = []
    negations = []
    
    for token in doc:
        if token.dep_ == "nsubj":  # Nominal subject
            subject = token.text
        elif token.dep_ == "ROOT":  # Main verb
            predicate = token.text
        elif token.dep_ in ["dobj", "attr", "pobj"]:  # Object
            object_ = token.text
        elif token.dep_ in ["amod", "advmod"]:  # Modifiers
            modifiers.append(token.text)
        elif token.dep_ == "neg":  # Negation
            negations.append(token.text)
    
    # Detect logical structures
    conditional_words = ["if", "when", "because", "since"]
    comparison_words = ["than", "compared", "versus", "more", "less"]
    
    has_conditional = any(token.text.lower() in conditional_words for token in doc)
    has_comparison = any(token.text.lower() in comparison_words for token in doc)
    has_negation = len(negations) > 0
    
    return {
        'subject': subject,
        'predicate': predicate,
        'object': object_,
        'modifiers': modifiers,
        'has_conditional': has_conditional,
        'has_comparison': has_comparison,
        'has_negation': has_negation,
        'full_parse': doc
    }
```

**Example**:

```
Input: "Earth is not the center of the solar system"

Output: {
    'subject': 'Earth',
    'predicate': 'is',
    'object': 'center',
    'modifiers': ['solar', 'system'],
    'has_conditional': False,
    'has_comparison': False,
    'has_negation': True
}
```

---

#### Step 2: Claim Type Classification

**Purpose**: Identify claim type to select appropriate decomposition strategy

**Algorithm**:

```python
def classify_claim_type(parsed_structure, claim_text, domain):
    """
    Classify claim type using pattern matching
    
    Returns:
        dict: {
            'primary_type': str,
            'all_types': list[str],
            'complexity': int
        }
    """
    
    claim_lower = claim_text.lower()
    claim_types = []
    
    # Pattern 1: Causal claim (X causes Y)
    causal_patterns = ["causes", "leads to", "results in", "because", "due to"]
    if any(pattern in claim_lower for pattern in causal_patterns):
        claim_types.append("causal")
    
    # Pattern 2: Categorical claim (X is a Y)
    if parsed_structure['predicate'] in ["is", "are", "was", "were", "am"]:
        claim_types.append("categorical")
    
    # Pattern 3: Relational claim (X relates to Y)
    relational_patterns = ["orbits", "surrounds", "contains", "includes", "part of"]
    if any(pattern in claim_lower for pattern in relational_patterns):
        claim_types.append("relational")
    
    # Pattern 4: Negation claim (X is not Y)
    if parsed_structure['has_negation']:
        claim_types.append("negation")
    
    # Pattern 5: Comparative claim (X is more/less than Y)
    if parsed_structure['has_comparison']:
        claim_types.append("comparative")
    
    # Pattern 6: Temporal claim (X happens before/after Y)
    temporal_patterns = ["before", "after", "during", "when", "while"]
    if any(pattern in claim_lower for pattern in temporal_patterns):
        claim_types.append("temporal")
    
    primary_type = claim_types[0] if claim_types else "simple_assertion"
    
    return {
        'primary_type': primary_type,
        'all_types': claim_types,
        'complexity': len(claim_types)
    }
```

---

#### Step 3: Domain Template System

**Purpose**: Use pre-defined decomposition templates for common domain patterns

**Template Structure**:

```python
DOMAIN_TEMPLATES = {
    'astronomy': {
        'orbital_claim': {
            'pattern': '{body} orbits {center}',
            'pattern_regex': r'(\w+)\s+orbits?\s+(\w+)',
            'variables': ['body', 'center'],
            
            'sub_claims': [
                '{body} exhibits periodic motion relative to {center}',
                '{body} shows apparent position changes against stellar background',
                'Gravitational forces between {body} and {center} cause orbital motion',
                '{body} distance to {center} varies in predictable pattern',
                '{body} velocity changes based on distance to {center}'
            ],
            
            'observable_tests': [
                'Measure {body} position over time relative to {center}',
                'Calculate orbital period of {body}',
                'Observe stellar parallax from {body} motion',
                'Measure {body} velocity at different orbital positions',
                'Calculate gravitational forces between {body} and {center}'
            ],
            
            'evidence_keywords': ['parallax', 'orbital', 'period', 'velocity', 'motion']
        },
        
        'centrality_claim': {
            'pattern': '{body} is center of {system}',
            'pattern_regex': r'(\w+)\s+(?:is|are)\s+(?:the\s+)?center\s+of\s+(?:the\s+)?(\w+)',
            'variables': ['body', 'system'],
            
            'sub_claims': [
                'All objects in {system} orbit {body}',
                '{body} does not exhibit orbital motion itself',
                '{body} is at gravitational center of mass of {system}',
                'Motion of {system} objects explained using {body} as reference frame',
                '{body} remains stationary relative to {system}'
            ],
            
            'observable_tests': [
                'Measure relative motions of all bodies in {system}',
                'Calculate center of mass of {system}',
                'Observe if {body} moves relative to stellar background',
                'Test if orbital mechanics work with {body} as center',
                'Measure gravitational influences within {system}'
            ],
            
            'evidence_keywords': ['center', 'stationary', 'orbit', 'reference', 'gravitational']
        }
    },
    
    'medical': {
        'treatment_efficacy': {
            'pattern': '{treatment} treats {condition}',
            'pattern_regex': r'(\w+)\s+treats?\s+(\w+)',
            'variables': ['treatment', 'condition'],
            
            'sub_claims': [
                '{treatment} reduces symptoms of {condition}',
                '{treatment} has biological mechanism relevant to {condition}',
                'Patients with {condition} improve after {treatment}',
                '{treatment} effects are specific to {condition}',
                '{treatment} outperforms placebo for {condition}'
            ],
            
            'observable_tests': [
                'Measure symptom severity before and after {treatment}',
                'Compare treated vs untreated patients with {condition}',
                'Verify biological mechanism of {treatment}',
                'Conduct placebo-controlled trials',
                'Measure dose-response relationship'
            ],
            
            'evidence_keywords': ['symptom', 'improvement', 'mechanism', 'trial', 'efficacy']
        }
    },
    
    'physics': {
        'force_relationship': {
            'pattern': '{force} acts on {object}',
            'pattern_regex': r'(\w+)\s+(?:acts|act)\s+on\s+(\w+)',
            'variables': ['force', 'object'],
            
            'sub_claims': [
                '{object} exhibits acceleration consistent with {force}',
                '{force} magnitude correlates with {object} motion',
                'Removing {force} changes {object} behavior',
                '{force} direction aligns with {object} motion direction',
                '{object} momentum changes when {force} is applied'
            ],
            
            'observable_tests': [
                'Measure {object} acceleration under {force}',
                'Vary {force} magnitude and observe effects on {object}',
                'Isolate {object} from {force} and observe behavior',
                'Perform vector analysis of {force} and motion',
                'Calculate momentum change of {object}'
            ],
            
            'evidence_keywords': ['acceleration', 'force', 'motion', 'momentum', 'vector']
        }
    }
}
```

**Template Matching Algorithm**:

```python
import re

def match_template(claim_text, domain_templates):
    """
    Find best matching template for claim
    
    Returns:
        tuple: (template_dict, extracted_variables) or (None, None)
    """
    
    best_template = None
    best_variables = None
    best_score = 0
    
    for template_name, template in domain_templates.items():
        # Try regex match
        match = re.search(template['pattern_regex'], claim_text, re.IGNORECASE)
        
        if match:
            # Extract variables from regex groups
            variables = {}
            for i, var_name in enumerate(template['variables']):
                variables[var_name] = match.group(i + 1)
            
            # Calculate match quality (simple: length of match)
            match_score = len(match.group(0)) / len(claim_text)
            
            if match_score > best_score:
                best_score = match_score
                best_template = template
                best_variables = variables
    
    return best_template, best_variables

def apply_template(template, variables):
    """
    Instantiate template with extracted variables
    
    Returns:
        dict: {
            'sub_claims': list[str],
            'observable_tests': list[str],
            'evidence_keywords': list[str]
        }
    """
    
    # Substitute variables in sub-claims
    sub_claims = []
    for sub_claim_template in template['sub_claims']:
        instantiated = sub_claim_template
        for var_name, var_value in variables.items():
            instantiated = instantiated.replace(f'{{{var_name}}}', var_value)
        sub_claims.append(instantiated)
    
    # Substitute variables in tests
    observable_tests = []
    for test_template in template['observable_tests']:
        instantiated = test_template
        for var_name, var_value in variables.items():
            instantiated = instantiated.replace(f'{{{var_name}}}', var_value)
        observable_tests.append(instantiated)
    
    return {
        'sub_claims': sub_claims,
        'observable_tests': observable_tests,
        'evidence_keywords': template['evidence_keywords']
    }
```

---

#### Step 4: Generic Decomposition (Fallback)

**Purpose**: When no template matches, use generic logical decomposition

**Algorithm**:

```python
def generic_decomposition(claim_text, parsed_structure, claim_type):
    """
    Generic decomposition based on claim type when no template matches
    
    Returns:
        dict: {
            'sub_claims': list[str],
            'observable_tests': list[str],
            'evidence_keywords': list[str]
        }
    """
    
    subject = parsed_structure['subject'] or "entity"
    predicate = parsed_structure['predicate'] or "relates to"
    object_ = parsed_structure['object'] or "property"
    
    primary_type = claim_type['primary_type']
    
    # Type-specific decomposition
    if primary_type == 'categorical':
        # "X is Y" decomposition
        sub_claims = [
            f"{subject} has defining properties of {object_}",
            f"{object_} category includes {subject}",
            f"{subject} behaves like other instances of {object_}",
            f"Classification tests identify {subject} as {object_}",
            f"{subject} shares characteristics with {object_}"
        ]
        
        observable_tests = [
            f"List defining properties of {object_}",
            f"Test if {subject} exhibits those properties",
            f"Compare {subject} to known {object_} examples",
            f"Apply classification criteria to {subject}",
            f"Measure similarity between {subject} and {object_}"
        ]
        
    elif primary_type == 'relational':
        # "X relates to Y" decomposition
        sub_claims = [
            f"Relationship between {subject} and {object_} exists",
            f"Relationship is measurable or observable",
            f"{subject} behavior depends on {object_}",
            f"Removing {object_} changes {subject} behavior",
            f"{subject} and {object_} influence each other"
        ]
        
        observable_tests = [
            f"Measure relationship strength between {subject} and {object_}",
            f"Observe {subject} in presence vs absence of {object_}",
            f"Vary {object_} and measure effect on {subject}",
            f"Test independence of {subject} from {object_}",
            f"Calculate correlation between {subject} and {object_}"
        ]
        
    elif primary_type == 'causal':
        # "X causes Y" decomposition
        sub_claims = [
            f"{subject} temporally precedes {object_}",
            f"{subject} correlates with {object_}",
            f"Mechanism exists linking {subject} to {object_}",
            f"Removing {subject} prevents {object_}",
            f"Varying {subject} predictably affects {object_}"
        ]
        
        observable_tests = [
            f"Establish temporal order: {subject} before {object_}",
            f"Measure correlation between {subject} and {object_}",
            f"Identify causal mechanism",
            f"Test if removing {subject} eliminates {object_}",
            f"Vary {subject} and measure {object_} response"
        ]
        
    elif primary_type == 'negation':
        # "X is not Y" decomposition
        sub_claims = [
            f"{subject} lacks defining properties of {object_}",
            f"{subject} behaves differently than {object_}",
            f"Tests distinguish {subject} from {object_}",
            f"{subject} belongs to different category than {object_}",
            f"{subject} contradicts expectations for {object_}"
        ]
        
        observable_tests = [
            f"List properties that distinguish {subject} from {object_}",
            f"Test if {subject} exhibits properties of {object_}",
            f"Compare {subject} behavior to {object_} behavior",
            f"Apply tests that separate {subject} and {object_}",
            f"Measure dissimilarity between {subject} and {object_}"
        ]
        
    elif primary_type == 'comparative':
        # "X is more/less than Y" decomposition
        sub_claims = [
            f"{subject} and {object_} are measurable on same scale",
            f"Measurement shows {subject} differs from {object_}",
            f"Difference is statistically significant",
            f"Comparison is consistent across measurements",
            f"Ordering relationship between {subject} and {object_} is stable"
        ]
        
        observable_tests = [
            f"Measure {subject} on relevant scale",
            f"Measure {object_} on same scale",
            f"Compare measurements statistically",
            f"Repeat measurements for consistency",
            f"Test if ordering holds under different conditions"
        ]
        
    else:  # simple_assertion or other
        # Generic decomposition
        sub_claims = [
            f"{subject} exhibits property: {predicate} {object_}",
            f"Evidence supports the claim about {subject}",
            f"Observations confirm {claim_text}",
            f"Tests verify the relationship",
            f"Multiple sources corroborate the claim"
        ]
        
        observable_tests = [
            f"Direct observation of {subject}",
            f"Measurement of relevant properties",
            f"Comparison with control cases",
            f"Experimental verification",
            f"Literature review for supporting evidence"
        ]
    
    # Generic evidence keywords
    evidence_keywords = [subject, object_, predicate, 'observation', 'measurement']
    
    return {
        'sub_claims': sub_claims,
        'observable_tests': observable_tests,
        'evidence_keywords': evidence_keywords
    }
```

---

#### Step 5: Antithesis Generation

**Purpose**: Generate the opposite claim and its sub-claims

**Algorithm**:

```python
def generate_antithesis(claim_text, parsed_structure, positive_decomposition):
    """
    Generate opposite claim and invert sub-claims
    
    Returns:
        dict: {
            'antithesis_text': str,
            'antithesis_sub_claims': list[str],
            'antithesis_tests': list[str]
        }
    """
    
    # Generate opposite claim text
    if parsed_structure['has_negation']:
        # Claim is negative, antithesis is positive
        antithesis_text = remove_negation(claim_text)
    else:
        # Claim is positive, antithesis is negative
        antithesis_text = add_negation(claim_text, parsed_structure)
    
    # Invert each sub-claim
    antithesis_sub_claims = [
        invert_claim(sub_claim) 
        for sub_claim in positive_decomposition['sub_claims']
    ]
    
    # Invert observable tests
    antithesis_tests = [
        invert_test(test)
        for test in positive_decomposition['observable_tests']
    ]
    
    return {
        'antithesis_text': antithesis_text,
        'antithesis_sub_claims': antithesis_sub_claims,
        'antithesis_tests': antithesis_tests
    }

def remove_negation(claim_text):
    """Remove negation from claim"""
    # Simple pattern matching for common negations
    claim_text = re.sub(r'\s+not\s+', ' ', claim_text, flags=re.IGNORECASE)
    claim_text = re.sub(r'\s+never\s+', ' always ', claim_text, flags=re.IGNORECASE)
    claim_text = re.sub(r'\s+no\s+', ' ', claim_text, flags=re.IGNORECASE)
    claim_text = re.sub(r"n't\s+", ' ', claim_text, flags=re.IGNORECASE)
    return claim_text.strip()

def add_negation(claim_text, parsed_structure):
    """Add negation to claim"""
    predicate = parsed_structure['predicate']
    
    if predicate:
        # Insert 'not' after verb
        pattern = r'\b' + re.escape(predicate) + r'\b'
        claim_text = re.sub(pattern, f"{predicate} not", claim_text, count=1)
    else:
        # Fallback: add 'not' after first verb-like word
        words = claim_text.split()
        for i, word in enumerate(words):
            if word.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'does', 'do']:
                words.insert(i + 1, 'not')
                break
        claim_text = ' '.join(words)
    
    return claim_text

def invert_claim(sub_claim):
    """Invert a sub-claim logically"""
    negation_words = ['not', 'no', 'never', 'lacks', 'without', 'fails']
    
    # Check if already negative
    has_negation = any(word in sub_claim.lower() for word in negation_words)
    
    if has_negation:
        # Remove negation to make positive
        for word in negation_words:
            sub_claim = re.sub(r'\b' + word + r'\b', '', sub_claim, flags=re.IGNORECASE)
        # Clean up extra spaces
        sub_claim = re.sub(r'\s+', ' ', sub_claim).strip()
    else:
        # Add negation to make negative
        # Find first verb and insert 'not' after it
        words = sub_claim.split()
        for i, word in enumerate(words):
            if word.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'does', 'do', 
                              'exhibits', 'shows', 'demonstrates', 'includes', 'contains']:
                words.insert(i + 1, 'not')
                break
        sub_claim = ' '.join(words)
    
    return sub_claim

def invert_test(test):
    """Invert an observable test"""
    # Tests often start with verbs like "Measure", "Observe", "Test"
    # For inversion, we change what we're looking for
    
    # If test looks for presence, look for absence
    if 'presence' in test.lower():
        test = test.replace('presence', 'absence')
    elif 'absence' in test.lower():
        test = test.replace('absence', 'presence')
    
    # If test expects to observe X, expect to NOT observe X
    if 'observe' in test.lower() and 'not' not in test.lower():
        test = test.replace('Observe', 'Observe absence of')
    
    return test
```

---

#### Step 6: Complete Decomposition Pipeline

**Main Function**:

```python
def decompose_claim(claim_text, domain, expert_metadata):
    """
    Main claim decomposition function
    
    Args:
        claim_text: Contradictory claim to decompose
        domain: Domain identifier (e.g., 'astronomy', 'medical')
        expert_metadata: Pre-computed metadata including templates
    
    Returns:
        dict: Complete structured decomposition
    """
    
    # Step 1: Parse syntactic structure
    parsed = parse_claim_structure(claim_text)
    
    # Step 2: Classify claim type
    claim_type = classify_claim_type(parsed, claim_text, domain)
    
    # Step 3: Load domain templates
    domain_templates = expert_metadata.get('layer2_templates', {}).get(domain, {})
    
    # Step 4: Try template matching
    template, variables = match_template(claim_text, domain_templates)
    
    if template and variables:
        # Use domain template
        positive_decomposition = apply_template(template, variables)
        decomposition_method = f"template:{template.get('pattern', 'unknown')}"
    else:
        # Fall back to generic decomposition
        positive_decomposition = generic_decomposition(claim_text, parsed, claim_type)
        decomposition_method = f"generic:{claim_type['primary_type']}"
    
    # Step 5: Generate antithesis
    antithesis = generate_antithesis(claim_text, parsed, positive_decomposition)
    
    # Step 6: Structure complete output
    return {
        'core_claim': claim_text,
        'decomposition_method': decomposition_method,
        
        'parsed_structure': {
            'subject': parsed['subject'],
            'predicate': parsed['predicate'],
            'object': parsed['object'],
            'modifiers': parsed['modifiers'],
        },
        
        'claim_type': claim_type,
        
        'positive_hypothesis': {
            'main_claim': claim_text,
            'supporting_claims': positive_decomposition['sub_claims'],
            'testable_assertions': positive_decomposition['observable_tests'],
            'evidence_keywords': positive_decomposition['evidence_keywords']
        },
        
        'negative_hypothesis': {
            'main_claim': antithesis['antithesis_text'],
            'supporting_claims': antithesis['antithesis_sub_claims'],
            'testable_assertions': antithesis['antithesis_tests'],
            'evidence_keywords': positive_decomposition['evidence_keywords']  # Same keywords
        },
        
        'variables': variables if variables else {},
    }
```

---

### Pre-Computation Requirements

**During Expert Training**, prepare domain templates:

```python
def prepare_layer2_metadata(domain, training_data):
    """
    Prepare decomposition templates for domain
    Can be done manually or semi-automatically
    """
    
    # Option 1: Manual template creation (recommended for initial deployment)
    templates = load_manual_templates(domain)
    
    # Option 2: Semi-automatic template extraction from training data
    # common_patterns = extract_claim_patterns(training_data)
    # templates = create_templates_from_patterns(common_patterns)
    
    metadata = {
        'layer2_templates': {
            domain: templates
        },
        'spacy_model': 'en_core_web_sm'  # Model version for compatibility
    }
    
    return metadata

def load_manual_templates(domain):
    """
    Load pre-authored templates
    These should be created by domain experts
    """
    # Return DOMAIN_TEMPLATES[domain] from template library
    return DOMAIN_TEMPLATES.get(domain, {})
```

---

### Performance Profile

|Operation|Time|Memory|Notes|
|---|---|---|---|
|spaCy parsing|~15ms|~50MB|One-time model load|
|Claim type classification|~2ms|Negligible|Pattern matching|
|Template matching|~5ms|~5MB|Regex + substitution|
|Generic decomposition|~10ms|Negligible|Fallback logic|
|Antithesis generation|~5ms|Negligible|Text manipulation|
|**Total Layer 2**|**~40ms**|**~55MB**|Per decomposition|

**Optimization opportunities**:

- Cache spaCy model loading (load once at startup)
- Pre-compile regex patterns in templates
- Parallelize sub-claim generation

---

## Integration Architecture

### Complete Pipeline Flow

```python
def reasoning_pipeline_layers_1_2(
    input_text,
    input_embedding,
    precheck_result,
    expert_metadata,
    domain
):
    """
    Complete Layers 1-2 pipeline
    
    Returns:
        dict: {
            'layer1_result': dict,
            'layer2_result': dict or None,
            'should_continue_reasoning': bool
        }
    """
    
    # === LAYER 1: Contradiction Analysis ===
    layer1_result = classify_contradiction_type(
        input_text,
        input_embedding,
        precheck_result,
        expert_metadata
    )
    
    # === DECISION POINT ===
    if not layer1_result['activation_decision']:
        # No contradiction or noise/unknown - skip Layer 2
        return {
            'layer1_result': layer1_result,
            'layer2_result': None,
            'should_continue_reasoning': False,
            'routing_decision': determine_routing(layer1_result, precheck_result)
        }
    
    # === LAYER 2: Claim Decomposition ===
    layer2_result = decompose_claim(
        input_text,
        domain,
        expert_metadata
    )
    
    return {
        'layer1_result': layer1_result,
        'layer2_result': layer2_result,
        'should_continue_reasoning': True,  # Continue to Layer 3
        'routing_decision': 'ACTIVATE_REASONING_MODE'
    }

def determine_routing(layer1_result, precheck_result):
    """
    Determine routing when reasoning mode not activated
    """
    
    if layer1_result['type'] == 'NOISE':
        return 'REJECT_INPUT'
    elif layer1_result['type'] == 'UNKNOWN':
        return 'CREATE_NEW_EXPERT'
    elif layer1_result['type'] == 'ALIGNED':
        return 'USE_EXISTING_EXPERT'
    elif layer1_result['type'] == 'PARTIALLY_CONTRADICTS':
        return 'CREATE_PATCH_WITH_CAUTION'
    else:
        return 'DEFAULT_TO_PRECHECK'
```

---

### Error Handling

```python
class ReasoningLayerError(Exception):
    """Base exception for reasoning layers"""
    pass

class Layer1Error(ReasoningLayerError):
    """Layer 1 specific errors"""
    pass

class Layer2Error(ReasoningLayerError):
    """Layer 2 specific errors"""
    pass

def safe_layer1_execution(input_text, input_embedding, precheck_result, expert_metadata):
    """
    Layer 1 with error handling and fallback
    """
    try:
        return classify_contradiction_type(
            input_text,
            input_embedding,
            precheck_result,
            expert_metadata
        )
    except Exception as e:
        logging.error(f"Layer 1 error: {e}")
        # Fallback: use pre-check result
        return {
            'type': 'ERROR',
            'contradiction_score': 0.0,
            'activation_decision': False,
            'error': str(e),
            'fallback_to_precheck': True
        }

def safe_layer2_execution(claim_text, domain, expert_metadata):
    """
    Layer 2 with error handling and fallback
    """
    try:
        return decompose_claim(claim_text, domain, expert_metadata)
    except Exception as e:
        logging.error(f"Layer 2 error: {e}")
        # Fallback: simple decomposition
        return {
            'core_claim': claim_text,
            'decomposition_method': 'fallback_error',
            'positive_hypothesis': {
                'main_claim': claim_text,
                'supporting_claims': [claim_text],
                'testable_assertions': ['Verify claim through evidence'],
                'evidence_keywords': []
            },
            'negative_hypothesis': {
                'main_claim': f"NOT({claim_text})",
                'supporting_claims': [f"NOT({claim_text})"],
                'testable_assertions': ['Verify opposite through evidence'],
                'evidence_keywords': []
            },
            'error': str(e)
        }
```

---

## Performance Requirements

### Latency Targets

|Component|Target|Maximum|Notes|
|---|---|---|---|
|Layer 1 total|25ms|50ms|Per input|
|Layer 2 total|40ms|80ms|When activated|
|Combined L1+L2|65ms|150ms|Before Layer 3|

### Memory Footprint

|Component|RAM|Storage|Notes|
|---|---|---|---|
|Layer 1 metadata|~15MB|~50MB|Per expert|
|Layer 2 templates|~5MB|~20MB|Per domain|
|spaCy model|~50MB|~15MB|Shared across domains|
|**Total per expert**|**~70MB**|**~85MB**|Loaded in memory|

### Scalability

- **Concurrent requests**: Layers 1-2 are stateless, can process in parallel
- **Multi-domain**: Each domain has separate metadata, loaded on-demand
- **Caching**: Template matching results can be cached for repeated claims

---

## Testing Strategy

### Unit Tests

**Layer 1 Tests**:

```python
def test_layer1_noise_detection():
    """Test Layer 1 correctly identifies noise"""
    test_cases = [
        "asdfasdf jkljkl",
        "!!!###$$",
        "aaaaaaaaaa",
        ""
    ]
    
    for test_input in test_cases:
        result = classify_contradiction_type(test_input, ...)
        assert result['type'] == 'NOISE', f"Failed on: {test_input}"

def test_layer1_unknown_detection():
    """Test Layer 1 correctly identifies unknown topics"""
    # Test with embeddings far from all training clusters
    unknown_embedding = generate_random_embedding()
    result = classify_contradiction_type("quantum entanglement", unknown_embedding, ...)
    assert result['type'] == 'UNKNOWN'

def test_layer1_contradiction_detection():
    """Test Layer 1 correctly identifies contradictions"""
    # For astronomy domain trained on geocentric model
    result = classify_contradiction_type("Earth orbits the Sun", ...)
    assert result['type'] in ['DIRECTLY_CONTRADICTS', 'PARTIALLY_CONTRADICTS']
    assert result['contradiction_score'] > 0.6
```

**Layer 2 Tests**:

```python
def test_layer2_template_matching():
    """Test template matching works correctly"""
    claim = "Earth orbits the Sun"
    result = decompose_claim(claim, 'astronomy', ...)
    
    assert 'orbital' in result['decomposition_method']
    assert len(result['positive_hypothesis']['supporting_claims']) > 0
    assert len(result['negative_hypothesis']['supporting_claims']) > 0

def test_layer2_generic_fallback():
    """Test generic decomposition works for unknown patterns"""
    claim = "Quantum computers utilize superposition"
    result = decompose_claim(claim, 'physics', ...)
    
    assert 'generic' in result['decomposition_method']
    assert len(result['positive_hypothesis']['supporting_claims']) > 0

def test_layer2_antithesis_generation():
    """Test antithesis generation inverts claims correctly"""
    claim = "Earth is not the center"
    result = decompose_claim(claim, 'astronomy', ...)
    
    antithesis = result['negative_hypothesis']['main_claim']
    assert 'not' not in antithesis.lower() or 'is the center' in antithesis.lower()
```

### Integration Tests

```python
def test_layers_1_2_integration():
    """Test full Layers 1-2 pipeline"""
    
    # Test case: Contradiction should activate Layer 2
    input_text = "Earth orbits the Sun"
    result = reasoning_pipeline_layers_1_2(input_text, ...)
    
    assert result['layer1_result']['type'] == 'DIRECTLY_CONTRADICTS'
    assert result['layer2_result'] is not None
    assert result['should_continue_reasoning'] == True

def test_layers_1_2_skip_on_noise():
    """Test Layer 2 skipped for noise"""
    
    input_text = "asdfasdf"
    result = reasoning_pipeline_layers_1_2(input_text, ...)
    
    assert result['layer1_result']['type'] == 'NOISE'
    assert result['layer2_result'] is None
    assert result['should_continue_reasoning'] == False
```

### Performance Tests

```python
import time

def test_layer1_performance():
    """Test Layer 1 meets latency requirements"""
    
    test_inputs = generate_test_inputs(n=100)
    
    start = time.time()
    for input_text, embedding, precheck in test_inputs:
        classify_contradiction_type(input_text, embedding, precheck, ...)
    end = time.time()
    
    avg_latency = (end - start) / 100
    assert avg_latency < 0.05, f"Layer 1 too slow: {avg_latency*1000}ms"

def test_layer2_performance():
    """Test Layer 2 meets latency requirements"""
    
    test_claims = generate_test_claims(n=100)
    
    start = time.time()
    for claim in test_claims:
        decompose_claim(claim, 'astronomy', ...)
    end = time.time()
    
    avg_latency = (end - start) / 100
    assert avg_latency < 0.08, f"Layer 2 too slow: {avg_latency*1000}ms"
```

---

## Deployment Considerations

### Environment Setup

**Dependencies**:

```
numpy>=1.21.0
scikit-learn>=1.0.0
spacy>=3.0.0
python>=3.8
```

**Model Downloads**:

```bash
python -m spacy download en_core_web_sm
```

### Configuration

```python
LAYER1_CONFIG = {
    'weights': {
        'embedding': 0.35,
        'ood': 0.25,
        'training': 0.25,
        'vocabulary': 0.15
    },
    'thresholds': {
        'noise': 0.6,
        'unknown': 0.6,
        'contradiction': 0.6
    },
    'enable_caching': True
}

LAYER2_CONFIG = {
    'spacy_model': 'en_core_web_sm',
    'enable_templates': True,
    'fallback_to_generic': True,
    'enable_caching': True,
    'max_sub_claims': 5,
    'max_tests': 5
}
```

### Monitoring

```python
def log_layer1_metrics(result):
    """Log Layer 1 performance metrics"""
    metrics = {
        'type': result['type'],
        'contradiction_score': result['contradiction_score'],
        'confidence': result['confidence'],
        'timestamp': time.time()
    }
    # Send to monitoring system
    monitoring.log('layer1.classification', metrics)

def log_layer2_metrics(result):
    """Log Layer 2 performance metrics"""
    metrics = {
        'decomposition_method': result['decomposition_method'],
        'num_sub_claims': len(result['positive_hypothesis']['supporting_claims']),
        'claim_type': result['claim_type']['primary_type'],
        'timestamp': time.time()
    }
    monitoring.log('layer2.decomposition', metrics)
```

---

## Future Enhancements

### Layer 1 Improvements

1. **Adaptive Thresholds**: Learn optimal thresholds per domain from validation data
2. **Additional Signals**: Incorporate linguistic features (sentiment, formality)
3. **Confidence Calibration**: Use Platt scaling or isotonic regression for confidence scores
4. **Multi-lingual Support**: Extend to non-English inputs

### Layer 2 Improvements

1. **Automated Template Discovery**: Mine common patterns from training data
2. **LLM-Assisted Decomposition**: Use small LLM (e.g., Phi-2) for complex claims as fallback
3. **Hierarchical Decomposition**: Support nested claims (claims within claims)
4. **Cross-Domain Templates**: Identify generalizable patterns across domains

---

## Appendix: Complete Example

### Input Scenario

```
Domain: astronomy
Expert: Trained on pre-1633 data (95% geocentric)
Input: "Earth is not the center of the solar system"
```

### Layer 1 Output

```json
{
  "type": "DIRECTLY_CONTRADICTS",
  "contradiction_score": 0.78,
  "dominant_pattern": "Geocentric model (Earth is center of universe)",
  "activation_decision": true,
  "confidence": 0.85,
  "reasoning": "Input strongly contradicts learned geocentric pattern",
  "signals": {
    "embedding": {
      "contradiction_score": 0.82,
      "dominant_similarity": -0.65,
      "opposing_similarity": 0.78
    },
    "ood": {
      "contradiction_indicator": 1.0,
      "svm_distance": 0.25
    },
    "training": {
      "is_opposing_region": true,
      "region_label": "heliocentric"
    },
    "vocabulary": {
      "contradiction_indicator": 0.71,
      "contradictory_tokens": ["not", "orbits", "sun"]
    }
  }
}
```

### Layer 2 Output

```json
{
  "core_claim": "Earth is not the center of the solar system",
  "decomposition_method": "template:centrality_claim",
  
  "parsed_structure": {
    "subject": "Earth",
    "predicate": "is",
    "object": "center",
    "modifiers": ["solar", "system"]
  },
  
  "claim_type": {
    "primary_type": "negation",
    "all_types": ["categorical", "negation"],
    "complexity": 2
  },
  
  "positive_hypothesis": {
    "main_claim": "Earth is not the center of the solar system",
    "supporting_claims": [
      "Not all objects in solar system orbit Earth",
      "Earth exhibits orbital motion itself",
      "Earth is not at gravitational center of mass of solar system",
      "Motion of solar system objects not explained using Earth as reference frame",
      "Earth does not remain stationary relative to solar system"
    ],
    "testable_assertions": [
      "Measure relative motions of all bodies in solar system",
      "Calculate center of mass of solar system",
      "Observe if Earth moves relative to stellar background",
      "Test if orbital mechanics work with Sun as center instead of Earth",
      "Measure gravitational influences within solar system"
    ],
    "evidence_keywords": ["center", "stationary", "orbit", "reference", "gravitational"]
  },
  
  "negative_hypothesis": {
    "main_claim": "Earth is the center of the solar system",
    "supporting_claims": [
      "All objects in solar system orbit Earth",
      "Earth does not exhibit orbital motion itself",
      "Earth is at gravitational center of mass of solar system",
      "Motion of solar system objects explained using Earth as reference frame",
      "Earth remains stationary relative to solar system"
    ],
    "testable_assertions": [
      "Measure relative motions showing all bodies orbit Earth",
      "Calculate center of mass at Earth's position",
      "Observe Earth remaining stationary relative to stellar background",
      "Test if orbital mechanics work with Earth as center",
      "Measure gravitational influences centered on Earth"
    ],
    "evidence_keywords": ["center", "stationary", "orbit", "reference", "gravitational"]
  },
  
  "variables": {
    "body": "Earth",
    "system": "solar system"
  }
}
```

### Pipeline Decision

```json
{
  "layer1_result": {...},
  "layer2_result": {...},
  "should_continue_reasoning": true,
  "routing_decision": "ACTIVATE_REASONING_MODE",
  "next_step": "Layer 3: Consequence Generator"
}
```

---

## Conclusion

Layers 1 and 2 establish the foundation for Mycelium's reasoning capability:

- **Layer 1** provides intelligent contradiction detection without expensive inference
- **Layer 2** structures contradictions into testable hypotheses

Both layers are designed to be:

- **Lightweight**: Combined latency < 150ms
- **No-inference**: Preserve pre-check layer optimization
- **Domain-agnostic**: Core algorithms work across domains with domain-specific extensions
- **Robust**: Graceful degradation and error handling
- **Testable**: Comprehensive unit and integration tests

These layers prepare structured input for Layers 3-6, which will evaluate evidence and synthesize reasoning.

---

## Document Changelog

|Version|Date|Changes|Author|
|---|---|---|---|
|1.0|2025-10-16|Initial specification|Mycelium Team|

---

## References

1. Layer 0 (Expert Pre-Check): Project Mycelium Progress Report 2
2. spaCy Documentation: https://spacy.io/usage/linguistic-features
3. Cosine Similarity: sklearn.metrics.pairwise.cosine_similarity
4. Relevant Logic Paper: "Why Cannot Large Language Models Ever Make True Correct Reasoning?" (arxiv.org/pdf/2508.10265)

---

**End of Layers 1-2 Implementation Specification**