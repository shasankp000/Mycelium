from enum import Enum


class DomainState(str, Enum):
    CREATING    = "CREATING"
    HOT         = "HOT"
    WARM        = "WARM"
    COLD        = "COLD"
    REMEMBERING = "REMEMBERING"
    DEPRECATED  = "DEPRECATED"
    ARCHIVED    = "ARCHIVED"


class DomainMode(str, Enum):
    BOOTSTRAP  = "BOOTSTRAP"   # head being trained from scratch
    EXPANSION  = "EXPANSION"   # head expanding to cover new sub-topics
    FROZEN     = "FROZEN"      # head locked, inference only
    THAWING    = "THAWING"     # temporary fine-tune window open


class GateState(str, Enum):
    OPEN     = "OPEN"     # domain accepting queries
    PENDING  = "PENDING"  # domain queued for reactivation
    CLOSED   = "CLOSED"   # domain in cold storage, not routing


class NoveltyDecision(str, Enum):
    ROUTE_EXISTING   = "ROUTE_EXISTING"   # query fits a known domain
    EXPAND_EXISTING  = "EXPAND_EXISTING"  # known domain needs sub-topic growth
    CREATE_NEW       = "CREATE_NEW"       # genuinely new domain needed
    DEFER_OOD        = "DEFER_OOD"        # below confidence floor, use OOD fallback
    REACTIVATE_COLD  = "REACTIVATE_COLD"  # query matches a cold domain, wake it


class DriftType(str, Enum):
    SEMANTIC    = "SEMANTIC"    # embedding centroid drift
    RETRIEVAL   = "RETRIEVAL"   # retrieval score degradation
    ROUTING     = "ROUTING"     # gate confidence decay
    CONFIDENCE  = "CONFIDENCE"  # head softmax entropy increase
    ACTIVATION  = "ACTIVATION"  # query volume drop (inactivity)
