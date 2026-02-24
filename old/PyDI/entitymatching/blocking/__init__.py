"""
Blocking subpackage: base interfaces and simple blocking strategies.
"""

import pandas as pd

from .base import BaseBlocker
from .noblocking import NoBlocker
from .standard import StandardBlocker
from .sorted_neighbourhood import SortedNeighbourhoodBlocker
from .token_blocking import TokenBlocker
from .embedding import EmbeddingBlocker

# Registry mapping blocker_type to class
_BLOCKER_REGISTRY: dict[str, type[BaseBlocker]] = {
    "TokenBlocker": TokenBlocker,
    "SortedNeighbourhoodBlocker": SortedNeighbourhoodBlocker,
    "EmbeddingBlocker": EmbeddingBlocker,
}


def blocker_from_spec(
    spec: dict,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
) -> BaseBlocker:
    """Create a blocker instance from a specification dict.

    Args:
        spec: A specification dict as returned by blocker.to_spec().
              Must contain a "blocker_type" key.
        df_left: Left dataset for blocking.
        df_right: Right dataset for blocking.
        id_column: Column name containing record IDs.

    Returns:
        A blocker instance of the appropriate type.

    Raises:
        ValueError: If the blocker_type is not recognized.
    """
    blocker_type = spec.get("blocker_type", "")
    if blocker_type not in _BLOCKER_REGISTRY:
        raise ValueError(
            f"Unknown blocker_type: {blocker_type!r}. "
            f"Available types: {list(_BLOCKER_REGISTRY.keys())}"
        )
    blocker_cls = _BLOCKER_REGISTRY[blocker_type]
    return blocker_cls.from_spec(spec, df_left, df_right, id_column)


__all__ = [
    "BaseBlocker",
    "NoBlocker",
    "StandardBlocker",
    "SortedNeighbourhoodBlocker",
    "TokenBlocker",
    "EmbeddingBlocker",
    "blocker_from_spec",
]
