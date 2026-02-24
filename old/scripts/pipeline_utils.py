"""
Pipeline utilities and configuration.

This module contains:
- Configuration loading and environment variable handling
- Cache checking functions
- Logging setup
- Helper functions for blocking/matching
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# Suppress sklearn, numpy, and xgboost warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# =============================================================================
# Configuration
# =============================================================================

class PipelineConfig:
    """Pipeline configuration loaded from environment variables."""

    def __init__(self):
        # Training/validation parameters
        self.training_embedding_threshold_cap = float(
            os.getenv("PYDI_TRAINING_EMBEDDING_THRESHOLD_CAP", "0.6")
        )
        self.max_training_candidates = int(
            os.getenv("PYDI_MAX_TRAINING_CANDIDATES", "25000")
        )
        self.label_batch_size = int(os.getenv("PYDI_LABEL_BATCH_SIZE", "25"))
        self.max_pairs_per_cluster = int(os.getenv("PYDI_MAX_PAIRS_PER_CLUSTER", "3"))

        # Blocking parameters
        self.force_blocking_reoptimize = os.getenv("PYDI_FORCE_BLOCKING_REOPTIMIZE", "0") == "1"
        self.reoptimize_blocking_with_training = os.getenv("PYDI_REOPTIMIZE_BLOCKING_WITH_TRAINING", "1") == "1"

        # Diagnostics/matching parameters
        self.diagnostics_max_candidates = int(
            os.getenv("PYDI_DIAGNOSTICS_MAX_CANDIDATES", "50000")
        )
        self.diagnostics_top_examples = int(
            os.getenv("PYDI_DIAGNOSTICS_TOP_EXAMPLES", "10")
        )

        # Fusion parameters
        self.fusion_max_candidates = int(
            os.getenv("PYDI_FUSION_MAX_CANDIDATES", "200000")
        )
        self.fusion_include_singletons = os.getenv("PYDI_FUSION_INCLUDE_SINGLETONS", "1") == "1"
        self.fusion_force_replan = os.getenv("PYDI_FUSION_FORCE_REPLAN", "0") == "1"

        # Active learning parameters
        self.target_positives = int(os.getenv("PYDI_TARGET_POSITIVES", "50"))
        self.target_negatives = int(os.getenv("PYDI_TARGET_NEGATIVES", "50"))
        self.max_total_labels = int(os.getenv("PYDI_MAX_TOTAL_LABELS", "5000"))
        self.labels_per_iteration = int(os.getenv("PYDI_LABELS_PER_ITERATION", "100"))
        self.active_learning_candidates = int(os.getenv("PYDI_ACTIVE_LEARNING_CANDIDATES", "5000"))
        self.max_iterations = int(os.getenv("PYDI_MAX_ITERATIONS", "30"))


# =============================================================================
# Directory Structure
# =============================================================================

class PipelineDirectories:
    """Manages pipeline output directory structure."""

    def __init__(self, output_dir: Path):
        self.output = output_dir
        self.schema_matching = output_dir / "schema_matching"
        self.entity_resolution = output_dir / "entity_resolution"
        self.validation = self.entity_resolution / "validation"
        self.training = self.entity_resolution / "training"
        self.blocking = self.entity_resolution / "blocking"
        self.matching = self.entity_resolution / "matching"
        self.test_evaluation = self.entity_resolution / "test_evaluation"
        self.fusion = output_dir / "fusion"
        self.fusion_validation = self.fusion / "validation"

    def create_all(self):
        """Create all directories."""
        for attr in dir(self):
            if not attr.startswith("_") and attr != "create_all":
                path = getattr(self, attr)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Cache Checking Functions
# =============================================================================

def check_schema_cache(data_dir: Path, output_dir: Path) -> bool:
    """Check if schema matching results are cached."""
    data_files = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))
    mappings_dir = output_dir / "mappings"
    return data_files and all(
        (mappings_dir / f"{f.stem}_mapping.csv").exists() and
        (output_dir / f"{f.stem}.csv").exists()
        for f in data_files
    )


def check_validation_cache(pairs: List[Tuple[str, str]], output_dir: Path) -> bool:
    """Check if all validation sets are cached."""
    return all(
        (output_dir / f"validation_{left}_{right}.csv").exists()
        for left, right in pairs
    )


def check_training_cache(pairs: List[Tuple[str, str]], output_dir: Path) -> bool:
    """Check if all training sets are cached."""
    return all(
        (output_dir / f"training_{left}_{right}_latest.csv").exists()
        for left, right in pairs
    )


def check_blocking_cache(
    pairs: List[Tuple[str, str]],
    cache: Dict,
    force_reoptimize: bool
) -> bool:
    """Check if all blocking configs are cached."""
    if force_reoptimize:
        return False
    if not isinstance(cache, dict):
        return False
    for left, right in pairs:
        key = f"{left}__{right}"
        cached = cache.get(key)
        if not isinstance(cached, dict) or not cached.get("blocker_spec") or not cached.get("blocking_columns"):
            return False
    return True


def check_matching_cache(pairs: List[Tuple[str, str]], output_dir: Path) -> bool:
    """Check if all matcher configs and correspondence files are cached."""
    config_path = output_dir / "matcher_configs.json"
    if not config_path.exists():
        return False
    try:
        configs = json.loads(config_path.read_text())
    except Exception:
        return False

    for left, right in pairs:
        key = f"{left}__{right}"
        if key not in configs:
            return False
        if not (output_dir / f"correspondences_{left}_{right}.csv").exists():
            return False
    return True


def check_fusion_cache(output_dir: Path) -> bool:
    """Check if fusion results are cached."""
    return (output_dir / "fused.csv").exists() and (output_dir / "fused_clean.csv").exists()


# =============================================================================
# Cache Loading and Saving
# =============================================================================

def load_json_cache(path: Path) -> Dict:
    """Load a JSON cache file, returning empty dict on failure."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def save_json_cache(path: Path, data: Dict) -> None:
    """Save data to a JSON cache file."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


# =============================================================================
# Blocker Spec Normalization
# =============================================================================

def normalize_blocker_spec(spec: Any) -> Any:
    """Best-effort normalization for cached blocker specs across versions."""
    if not isinstance(spec, dict):
        return spec

    out = dict(spec)
    blocker_type = out.get("blocker_type")

    # Legacy key rename: EmbeddingBlocker used to store "backend" instead of "index_backend".
    if blocker_type == "EmbeddingBlocker" and "backend" in out and "index_backend" not in out:
        out["index_backend"] = out.pop("backend")

    # Legacy shape: some configs stored "blocking_columns" for EmbeddingBlocker.
    if blocker_type == "EmbeddingBlocker" and "text_cols" not in out and "blocking_columns" in out:
        bc = out.get("blocking_columns")
        if isinstance(bc, list) and bc:
            out["text_cols"] = bc

    # Legacy shape: Token/SNB blockers stored a list under "blocking_columns".
    if blocker_type in {"TokenBlocker", "SortedNeighbourhoodBlocker"} and "blocking_column" not in out:
        bc = out.get("blocking_columns")
        if isinstance(bc, list) and len(bc) == 1 and isinstance(bc[0], str):
            out["blocking_column"] = bc[0]

    return out


def normalize_blocking_cache(cache: Any) -> Dict:
    """Normalize all entries in a blocking cache."""
    if not isinstance(cache, dict):
        return {}

    out: Dict = {}
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            out[key] = entry
            continue
        norm = dict(entry)
        norm["blocker_spec"] = normalize_blocker_spec(norm.get("blocker_spec"))
        out[key] = norm
    return out


# =============================================================================
# Logging Setup
# =============================================================================

class ConsoleFilter(logging.Filter):
    """Keep console output minimal: show progress + warnings/errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        # Only show LLM labeling progress on console.
        return record.name == "PyDI.entitymatching.llm_progress"


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # File handler (full logging)
    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console logging disabled - only file logging active


# =============================================================================
# Helper Functions
# =============================================================================

def pair_key(left: str, right: str) -> str:
    """Generate cache key for a dataset pair."""
    return f"{left}__{right}"


def collect_candidate_batches(
    blocker,
    *,
    limit: int
) -> Tuple[List[pd.DataFrame], int]:
    """Collect candidate batches from a blocker up to a limit."""
    batches: List[pd.DataFrame] = []
    total = 0
    for batch in blocker:
        if batch is None or batch.empty:
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        take = batch[["id1", "id2"]].dropna(subset=["id1", "id2"]).drop_duplicates().head(remaining)
        if take.empty:
            continue
        batches.append(take)
        total += len(take)
        if total >= limit:
            break
    return batches, total


def find_training_file(
    training_dir: Path,
    left_name: str,
    right_name: str
) -> Optional[Path]:
    """Find the best training file to use (prefer _latest, fall back to original)."""
    # First check for _latest file (from previous active learning run)
    latest_patterns = [
        f"training_{left_name}_{right_name}_latest.csv",
        f"training_{right_name}_{left_name}_latest.csv",
    ]
    for pattern in latest_patterns:
        candidate_path = training_dir / pattern
        if candidate_path.exists():
            return candidate_path

    # If no _latest, fall back to original FAISS/hybrid file
    original_patterns = [
        f"similarity_training_faiss_{left_name}_{right_name}.csv",
        f"similarity_training_faiss_{right_name}_{left_name}.csv",
        f"similarity_training_hybrid_{left_name}_{right_name}.csv",
        f"similarity_training_hybrid_{right_name}_{left_name}.csv",
    ]
    for pattern in original_patterns:
        candidate_path = training_dir / pattern
        if candidate_path.exists():
            return candidate_path

    return None
