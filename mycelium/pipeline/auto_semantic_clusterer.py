"""
Automatic Semantic Tag Clustering using Sentence Transformers
Eliminates manual dictionary maintenance by learning tag relationships from embeddings
"""

import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import json

class AutoSemanticClusterer:
    """
    Automatically clusters tags into domain groups using semantic embeddings.
    No manual dictionary maintenance required!
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_file=None):
        """
        Initialize the auto-clustering system.
        
        Args:
            model_name: Sentence transformer model to use
            cache_file: Where to cache computed embeddings
        """
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_name = model_name
        self.cache_file = cache_file if cache_file is not None else os.path.join(
            _project_root, 'runtime', 'cache', 'semantic_clusters_cache.pkl'
        )
        self.model = None
        
        # Core domain anchors — keys MUST match the canonical domain names used
        # by UnifiedExpertSystem (expert registry) and ExpertFilter.domain_list.
        # Previously 'mathematics' and 'computer_science' were used here, which
        # caused normalize_domain('maths') and normalize_domain('AI') to always
        # return None (no anchor match), producing false missing_domain entries.
        self.domain_anchors = {
            'medical': [
                'medicine', 'healthcare', 'biology', 'disease', 'treatment',
                'patient', 'clinical', 'diagnosis', 'therapy', 'medical research'
            ],
            'physics': [
                'physics', 'quantum mechanics', 'thermodynamics', 'particle physics',
                'electromagnetic', 'optics', 'mechanics', 'energy', 'force'
            ],
            'chemistry': [
                'chemistry', 'chemical reaction', 'molecule', 'compound', 'element',
                'organic chemistry', 'inorganic', 'biochemistry', 'catalyst'
            ],
            # Renamed from 'mathematics' -> 'maths' to match ExpertFilter.domain_list
            # and the expert registry key used throughout the codebase.
            'maths': [
                'mathematics', 'algebra', 'calculus', 'geometry', 'statistics',
                'equation', 'theorem', 'mathematical proof', 'number theory', 'maths'
            ],
            # Renamed from 'computer_science' -> 'AI' to match ExpertFilter.domain_list.
            'AI': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'NLP', 'computer science', 'programming',
                'algorithm', 'software', 'data structure'
            ],
            'music': [
                'music', 'song', 'melody', 'harmony', 'instrument', 'composition',
                'rhythm', 'performance', 'musical notation'
            ],
            # Additional anchors so every domain in ExpertFilter.domain_list is
            # covered and normalize_domain() never returns None for a valid tag.
            'technology': [
                'technology', 'engineering', 'hardware', 'digital', 'electronics',
                'robotics', 'automation', 'semiconductor', 'circuit'
            ],
            'literature': [
                'literature', 'novel', 'poetry', 'fiction', 'prose', 'author',
                'narrative', 'genre', 'literary analysis', 'writing'
            ],
            'history': [
                'history', 'historical event', 'civilization', 'war', 'empire',
                'archaeology', 'chronology', 'ancient', 'medieval', 'modern history'
            ],
            'sports': [
                'sports', 'athletics', 'football', 'basketball', 'tennis',
                'competition', 'training', 'athlete', 'tournament', 'fitness'
            ],
            'politics': [
                'politics', 'government', 'policy', 'election', 'democracy',
                'legislation', 'parliament', 'diplomacy', 'international relations'
            ],
            'finance': [
                'finance', 'economics', 'investment', 'banking', 'stock market',
                'currency', 'accounting', 'budget', 'financial analysis'
            ],
            'logistics': [
                'logistics', 'supply chain', 'transport', 'shipping', 'warehouse',
                'distribution', 'inventory', 'freight', 'procurement'
            ],
            'education': [
                'education', 'teaching', 'learning', 'curriculum', 'pedagogy',
                'school', 'university', 'student', 'academic', 'instruction'
            ],
            'art': [
                'art', 'painting', 'sculpture', 'drawing', 'design', 'aesthetic',
                'gallery', 'artist', 'visual art', 'creative'
            ],
        }
        
        # Similarity threshold for clustering.
        # NOTE: ExpertFilter sets this via clusterer.similarity_threshold = X
        # after construction.  load_cache() deliberately does NOT restore this
        # value from the pickle so the caller always owns the threshold.
        self.similarity_threshold = 0.5  # default; overridden by ExpertFilter
        
        # Cache for tag -> domain mappings
        self.tag_to_domain = {}
        self.domain_embeddings = {}
        
    def _load_model(self):
        """Lazy-load the sentence transformer model via ModelRegistry.

        Delegates to model_registry.get_model() so the model is loaded
        once per process and served from the in-memory cache on every
        subsequent call.  device='cpu' is pinned explicitly to avoid
        competing with Ollama for GPU VRAM.

        Falls back to a direct SentenceTransformer() load only when
        model_registry is unavailable (stripped test environments).
        """
        if self.model is not None:
            return
        try:
            from model_registry import get_model
            self.model = get_model(
                self.model_name,
                model_type="sentence_transformer",
                device="cpu",
            )
        except ImportError:
            # model_registry not available — fall back to direct load
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device="cpu")
    
    def _compute_domain_embeddings(self):
        """Compute average embedding for each domain from its anchor terms."""
        self._load_model()
        
        print("\n🔧 Computing domain embeddings from anchor terms...")
        for domain, anchors in self.domain_anchors.items():
            embeddings = self.model.encode(anchors)
            domain_embedding = np.mean(embeddings, axis=0)
            self.domain_embeddings[domain] = domain_embedding
            print(f"   ✓ {domain}: {len(anchors)} anchor terms")
    
    def cluster_tag(self, tag):
        """
        Automatically determine which domain a tag belongs to.
        
        Args:
            tag: Tag string to cluster
            
        Returns:
            (domain, confidence) tuple, or (None, 0) if no good match
        """
        if tag in self.tag_to_domain:
            return self.tag_to_domain[tag], 1.0
        
        self._load_model()
        tag_embedding = self.model.encode([tag])[0]
        
        best_domain = None
        best_similarity = -1
        
        for domain, domain_emb in self.domain_embeddings.items():
            similarity = cosine_similarity([tag_embedding], [domain_emb])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_domain = domain
        
        if best_similarity >= self.similarity_threshold:
            self.tag_to_domain[tag] = best_domain
            return best_domain, float(best_similarity)
        else:
            return None, float(best_similarity)
    
    def cluster_tags_batch(self, tags):
        """
        Cluster multiple tags at once (more efficient).
        
        Args:
            tags: List of tag strings
            
        Returns:
            Dictionary mapping tag -> (domain, confidence)
        """
        self._load_model()
        
        uncached_tags = [t for t in tags if t not in self.tag_to_domain]
        
        if not uncached_tags:
            return {t: (self.tag_to_domain[t], 1.0) for t in tags}
        
        print(f"\n🔍 Clustering {len(uncached_tags)} new tags...")
        tag_embeddings = self.model.encode(uncached_tags)
        
        results = {}
        
        for tag, tag_emb in zip(uncached_tags, tag_embeddings):
            best_domain = None
            best_similarity = -1
            
            for domain, domain_emb in self.domain_embeddings.items():
                similarity = cosine_similarity([tag_emb], [domain_emb])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_domain = domain
            
            if best_similarity >= self.similarity_threshold:
                self.tag_to_domain[tag] = best_domain
                results[tag] = (best_domain, float(best_similarity))
            else:
                results[tag] = (None, float(best_similarity))
        
        for tag in tags:
            if tag in self.tag_to_domain and tag not in results:
                results[tag] = (self.tag_to_domain[tag], 1.0)
        
        return results
    
    def save_cache(self):
        """Save computed mappings to disk."""
        cache_data = {
            'tag_to_domain': self.tag_to_domain,
            'domain_embeddings': self.domain_embeddings,
            'domain_anchors': self.domain_anchors,
            # NOTE: similarity_threshold is intentionally NOT saved here.
            # The threshold is owned by the caller (ExpertFilter) and must
            # not be restored from a stale pickle on the next run.
        }
        
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"\n💾 Cached {len(self.tag_to_domain)} tag mappings to {self.cache_file}")
    
    def load_cache(self):
        """Load cached mappings from disk.
        
        NOTE: similarity_threshold is deliberately NOT restored from the cache.
        ExpertFilter sets it after construction; letting the pickle override it
        caused a threshold race where the cache would silently revert to 0.5
        even when ExpertFilter requested 0.45.
        """
        if not os.path.exists(self.cache_file):
            print(f"⚠️ No cache found at {self.cache_file}")
            return False
        
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.tag_to_domain = cache_data.get('tag_to_domain', {})
        self.domain_embeddings = cache_data.get('domain_embeddings', {})
        self.domain_anchors = cache_data.get('domain_anchors', self.domain_anchors)
        # similarity_threshold is intentionally NOT restored here — see docstring.
        
        print(f"✅ Loaded {len(self.tag_to_domain)} cached tag mappings")
        return True
    
    def initialize(self):
        """Initialize the clusterer (compute domain embeddings)."""
        if self.load_cache():
            # Recompute embeddings if the cached anchor set differs from the
            # current one (e.g. after renaming 'mathematics' -> 'maths').
            if set(self.domain_anchors.keys()) != set(
                d for d in self.domain_embeddings.keys()
            ):
                print("🔄 Anchor set changed — recomputing domain embeddings...")
                self._compute_domain_embeddings()
                self.save_cache()
            else:
                print("✅ Using cached embeddings")
        else:
            print("🔄 Computing domain embeddings from scratch...")
            self._compute_domain_embeddings()
            self.save_cache()
    
    def export_mappings_to_json(self, filepath='semantic_clusters.json'):
        """Export current tag->domain mappings to JSON for inspection."""
        domain_to_tags = {}
        for tag, domain in self.tag_to_domain.items():
            if domain not in domain_to_tags:
                domain_to_tags[domain] = []
            domain_to_tags[domain].append(tag)
        
        with open(filepath, 'w') as f:
            json.dump(domain_to_tags, f, indent=2)
        
        print(f"📄 Exported mappings to {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("AUTO SEMANTIC TAG CLUSTERING DEMO")
    print("="*80)
    
    clusterer = AutoSemanticClusterer()
    clusterer.initialize()
    
    test_tags = [
        'biology', 'healthcare', 'immunology', 'genetics', 'neuroscience',
        'quantum', 'electromagnetism', 'thermodynamics', 'relativity',
        'organic', 'biochemistry', 'polymer', 'catalyst',
        'calculus', 'topology', 'linear algebra', 'probability',
        'neural network', 'deep learning', 'algorithm', 'database',
        'piano', 'guitar', 'symphony', 'jazz'
    ]
    
    print(f"\n🧪 Testing {len(test_tags)} tags...")
    results = clusterer.cluster_tags_batch(test_tags)
    
    from collections import defaultdict
    by_domain = defaultdict(list)
    
    for tag, (domain, confidence) in results.items():
        by_domain[domain].append((tag, confidence))
    
    for domain, tag_list in sorted(by_domain.items(), key=lambda x: (x[0] is None, x[0] or '')):
        print(f"\n🔧 {domain.upper() if domain else 'UNMATCHED'}:")
        for tag, conf in sorted(tag_list, key=lambda x: x[1], reverse=True):
            print(f"   {tag:20s} (similarity: {conf:.3f})")
    
    clusterer.save_cache()
    clusterer.export_mappings_to_json()
    
    print("\n" + "="*80)
    print("✅ Auto-clustering complete!")
    print("="*80)
