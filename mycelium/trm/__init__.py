"""Phase D — TRM: Graph Store, DFS Lookup, Promotion Pipeline,
and Samsung TRM Neural Reasoner (arXiv:2510.04871).
"""

# --- Graph-persistence layer (Phase D §D.1) ---
from .graph_store import GraphStore
from .promotion import PromotionPolicy
from .dfs_lookup import DFSLookup, TRMLookupResult
from .trm_engine import TRMEngine

# --- Samsung TRM neural reasoner (Phase D §D.2) ---
from .config import TRMConfig
from .network import TRMCell
from .embeddings import QueryEmbedder, AnswerEmbedder
from .output_head import DomainHead, HaltHead
from .reasoner import TRMReasoner, TRMOutput
from .trainer import TRMTrainer
from .integration import TRMLens

__all__ = [
    # persistence
    "GraphStore",
    "PromotionPolicy",
    "DFSLookup",
    "TRMLookupResult",
    "TRMEngine",
    # neural reasoner
    "TRMConfig",
    "TRMCell",
    "QueryEmbedder",
    "AnswerEmbedder",
    "DomainHead",
    "HaltHead",
    "TRMReasoner",
    "TRMOutput",
    "TRMTrainer",
    "TRMLens",
]
