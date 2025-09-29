import json
import datetime
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.cluster import AgglomerativeClustering
import ollama
# Import dummy expert models
from layer_2_prototype import get_expert_model
def embed_tags_transformer(tags, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(tags)
    return embeddings

def cluster_tags_transformer(tags, embeddings, similarity_threshold=0.7):
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

def extract_tags_openai(text, api_key, endpoint="https://api.openai.com/v1/chat/completions", provider="openai", model="gpt-3.5-turbo"):
    import requests
    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags (choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\nSentence: {text}"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.2
    }
    response = requests.post(endpoint, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # OpenAI-compatible response parsing
    if provider == "openai" or provider == "openrouter" or provider == "perplexity":
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

DOMAIN_LIST = [
    "AI", "healthcare", "logistics", "finance", "education", "physics", "maths", "biology", "chemistry", "technology", "sports", "politics", "history", "art", "music", "literature"
]


def extract_tags_llama(text, model="llama3:8b"):
    prompt = (
        f"Analyze the following sentence and output ONLY a comma-separated list of domain tags (choose from: {', '.join(DOMAIN_LIST)}). Do not include any explanation, headers, or extra text.\nSentence: {text}"
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

def normalize_tags(tags, domain_list=DOMAIN_LIST, threshold=80):
    normalized = []
    for tag in tags:
        match, score = process.extractOne(tag, domain_list)
        normalized.append(match if score >= threshold else tag)
    return normalized

def cluster_tags(tags, embeddings, distance_threshold=0.3):
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

def save_clusters_to_json(clusters, text, extractor, embed_model, distance_threshold, filename="tag_clusters.json"):
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

def load_clusters_from_json(filename="tag_clusters.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# Temporal Locality Layer
class TemporalLocalityLayer:
    def __init__(self, max_size=100, time_window_hours=24):
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
    
    def get_recent_statements(self, time_limit_hours=None):
        """Get recent statements within time window"""
        if time_limit_hours is None:
            return list(self.recent_statements)
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=time_limit_hours)
        recent = [entry for entry in self.recent_statements 
                 if entry["parsed_time"] >= cutoff_time]
        return recent
    
    def get_temporal_similarity(self, input_tags, time_limit_hours=1):
        """Calculate temporal similarity based on recent tag overlap"""
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

def assign_domain_patch(spatial_analysis, similarity_threshold=0.3):
    """Assign domain/patch network based on spatial locality analysis"""
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
