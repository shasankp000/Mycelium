# mycelium/pipeline/predicates/__init__.py
# Public surface of the predicate subsystem.
#
# Everything a caller needs for the full pipeline is importable from here:
#
#   from mycelium.pipeline.predicates import get_pipeline, PredicateStore
#   store = get_pipeline().run(text, domain_hint='science')

from mycelium.pipeline.predicates.predicate_extractor import (
    PredicateExtractor,
    get_extractor,
)
from mycelium.pipeline.predicates.predicate_negator import (
    PredicateNegator,
    get_negator,
)
from mycelium.pipeline.predicates.predicate_pipeline import (
    PredicatePipeline,
    get_pipeline,
)
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    PredicateFrame,
    PredicateType,
)

__all__ = [
    # Pipeline facade
    "PredicatePipeline",
    "get_pipeline",
    # Extractor
    "PredicateExtractor",
    "get_extractor",
    # Negator
    "PredicateNegator",
    "get_negator",
    # Store
    "PredicateStore",
    # Types
    "PredicateFrame",
    "PredicateType",
]
