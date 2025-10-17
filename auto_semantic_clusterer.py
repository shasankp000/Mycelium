"""
Automatic Semantic Tag Clustering using Sentence Transformers
Eliminates manual dictionary maintenance by learning tag relationships from embeddings
"""

import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

class AutoSemanticClusterer:
    """
    Automatically clusters tags into domain groups using semantic embeddings.
    No manual dictionary maintenance required!
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_file='semantic_clusters_cache.pkl'):
        """
        Initialize the auto-clustering system.
        
        Args:
            model_name: Sentence transformer model to use
            cache_file: Where to cache computed embeddings
        """
        self.model_name = model_name
        self.cache_file = cache_file
        self.model = None
        
        # Core domain anchors - these define the canonical domains we care about
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
            'mathematics': [
                'mathematics', 'algebra', 'calculus', 'geometry', 'statistics',
                'equation', 'theorem', 'mathematical proof', 'number theory'
            ],
            'computer_science': [
                'computer science', 'programming', 'algorithm', 'software', 'hardware',
                'artificial intelligence', 'machine learning', 'data structure'
            ],
            'music': [
                'music', 'song', 'melody', 'harmony', 'instrument', 'composition',
                'rhythm', 'performance', 'musical notation'
            ]
        }
        
        # Similarity threshold for clustering
        self.similarity_threshold = 0.5  # Adjustable
        
        # Cache for tag -> domain mappings
        self.tag_to_domain = {}
        self.domain_embeddings = {}
        
    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            print(f"📥 Loading semantic model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded")
    
    def _compute_domain_embeddings(self):
        """Compute average embedding for each domain from its anchor terms."""
        self._load_model()
        
        print("\n🔧 Computing domain embeddings from anchor terms...")
        for domain, anchors in self.domain_anchors.items():
            # Get embeddings for all anchor terms
            embeddings = self.model.encode(anchors)
            # Average them to get domain centroid
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
        # Check cache first
        if tag in self.tag_to_domain:
            return self.tag_to_domain[tag], 1.0
        
        # Compute tag embedding
        self._load_model()
        tag_embedding = self.model.encode([tag])[0]
        
        # Compare to all domain embeddings
        best_domain = None
        best_similarity = -1
        
        for domain, domain_emb in self.domain_embeddings.items():
            similarity = cosine_similarity([tag_embedding], [domain_emb])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_domain = domain
        
        # Only return if similarity exceeds threshold
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
        
        # Filter out cached tags
        uncached_tags = [t for t in tags if t not in self.tag_to_domain]
        
        if not uncached_tags:
            # All cached
            return {t: (self.tag_to_domain[t], 1.0) for t in tags}
        
        # Compute embeddings for uncached tags
        print(f"\n🔍 Clustering {len(uncached_tags)} new tags...")
        tag_embeddings = self.model.encode(uncached_tags)
        
        results = {}
        
        # Process each tag
        for tag, tag_emb in zip(uncached_tags, tag_embeddings):
            best_domain = None
            best_similarity = -1
            
            for domain, domain_emb in self.domain_embeddings.items():
                similarity = cosine_similarity([tag_emb], [domain_emb])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_domain = domain
            
            # Cache and return
            if best_similarity >= self.similarity_threshold:
                self.tag_to_domain[tag] = best_domain
                results[tag] = (best_domain, float(best_similarity))
            else:
                results[tag] = (None, float(best_similarity))
        
        # Add cached results
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
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"\n💾 Cached {len(self.tag_to_domain)} tag mappings to {self.cache_file}")
    
    def load_cache(self):
        """Load cached mappings from disk."""
        if not os.path.exists(self.cache_file):
            print(f"⚠️ No cache found at {self.cache_file}")
            return False
        
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.tag_to_domain = cache_data.get('tag_to_domain', {})
        self.domain_embeddings = cache_data.get('domain_embeddings', {})
        self.domain_anchors = cache_data.get('domain_anchors', self.domain_anchors)
        self.similarity_threshold = cache_data.get('similarity_threshold', self.similarity_threshold)
        
        print(f"✅ Loaded {len(self.tag_to_domain)} cached tag mappings")
        return True
    
    def initialize(self):
        """Initialize the clusterer (compute domain embeddings)."""
        # Try loading cache first
        if self.load_cache():
            print("✅ Using cached embeddings")
        else:
            print("🔄 Computing domain embeddings from scratch...")
            self._compute_domain_embeddings()
            self.save_cache()
    
    def export_mappings_to_json(self, filepath='semantic_clusters.json'):
        """Export current tag->domain mappings to JSON for inspection."""
        # Group by domain
        domain_to_tags = {}
        for tag, domain in self.tag_to_domain.items():
            if domain not in domain_to_tags:
                domain_to_tags[domain] = []
            domain_to_tags[domain].append(tag)
        
        with open(filepath, 'w') as f:
            json.dump(domain_to_tags, f, indent=2)
        
        print(f"📄 Exported mappings to {filepath}")


if __name__ == "__main__":
    # Demo
    print("="*80)
    print("AUTO SEMANTIC TAG CLUSTERING DEMO")
    print("="*80)
    
    clusterer = AutoSemanticClusterer()
    clusterer.initialize()
    
    # Test with various tags
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
    
    # Print results grouped by domain
    from collections import defaultdict
    by_domain = defaultdict(list)
    
    for tag, (domain, confidence) in results.items():
        by_domain[domain].append((tag, confidence))
    
    for domain, tag_list in sorted(by_domain.items(), key=lambda x: (x[0] is None, x[0] or '')):
        print(f"\n🔧 {domain.upper() if domain else 'UNMATCHED'}:")
        for tag, conf in sorted(tag_list, key=lambda x: x[1], reverse=True):
            print(f"   {tag:20s} (similarity: {conf:.3f})")
    
    # Save cache
    clusterer.save_cache()
    clusterer.export_mappings_to_json()
    
    print("\n" + "="*80)
    print("✅ Auto-clustering complete!")
    print("   Now tags are automatically mapped to domains using semantic similarity!")
    print("="*80)
