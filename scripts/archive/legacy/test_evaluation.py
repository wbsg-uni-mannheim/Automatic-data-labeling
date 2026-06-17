"""Test set evaluation for entity resolution pipelines.

This module provides functions to evaluate entity matching and data fusion
against held-out test sets.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _create_feature_extractor(df_left: pd.DataFrame, df_right: pd.DataFrame, id_column: str = "id"):
    """Create a FeatureExtractor based on common columns between datasets.

    Uses heuristics to determine appropriate comparators for each column type.
    """
    from PyDI.entitymatching.comparators import StringComparator, NumericComparator, DateComparator
    from PyDI.entitymatching.feature_extraction import FeatureExtractor

    # Find common columns (excluding id column)
    left_cols = set(df_left.columns) - {id_column}
    right_cols = set(df_right.columns) - {id_column}
    common_cols = left_cols & right_cols

    if not common_cols:
        raise ValueError("No common columns found between datasets")

    comparators = []

    for col in sorted(common_cols):
        # Sample values from both datasets to determine type
        left_sample = df_left[col].dropna().head(10)
        right_sample = df_right[col].dropna().head(10)

        # Check if column contains lists
        has_lists = False
        for val in list(left_sample) + list(right_sample):
            if isinstance(val, (list, tuple, np.ndarray)):
                has_lists = True
                break

        list_strategy = "concatenate" if has_lists else None

        # Check if numeric
        try:
            if left_sample.dtype in [np.float64, np.int64, float, int]:
                comparators.append(NumericComparator(col, "relative_difference", max_difference=0.5, list_strategy=list_strategy))
                continue
        except:
            pass

        # Check if date-like
        if 'date' in col.lower() or 'year' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(left_sample.iloc[0] if len(left_sample) > 0 else "")
                comparators.append(DateComparator(col, max_days_difference=365, list_strategy="closest_dates" if has_lists else None))
                continue
            except:
                pass

        # Default to string comparator
        comparators.append(StringComparator(col, "jaro_winkler", list_strategy=list_strategy))

    return FeatureExtractor(comparators)


def train_matcher_from_file(
    train_file: Path,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    matcher_type: str,
    id_column: str = "id",
) -> tuple[Any, Any]:
    """Train a matcher using user-provided training file.

    Parameters
    ----------
    train_file : Path
        CSV with columns: id1, id2, label (TRUE/FALSE or true/false)
    df_left, df_right : pd.DataFrame
        Source dataframes for feature extraction
    matcher_type : str
        One of: "logreg", "rf", "gb", "xgb"
    id_column : str
        Name of ID column

    Returns
    -------
    tuple[classifier, feature_extractor]
        Trained classifier and feature extractor
    """
    # Load training data - detect if file has header
    # First read a few lines to check
    with open(train_file, 'r') as f:
        first_line = f.readline().strip()

    # Check if first line looks like a header (contains "id1" or "label")
    if 'id1' in first_line.lower() or 'label' in first_line.lower():
        train_df = pd.read_csv(train_file)
        # Ensure column names are lowercase
        train_df.columns = [c.lower() for c in train_df.columns]
    else:
        train_df = pd.read_csv(train_file, header=None, names=["id1", "id2", "label"])

    train_df["id1"] = train_df["id1"].astype(str)
    train_df["id2"] = train_df["id2"].astype(str)
    train_df["label"] = train_df["label"].astype(str).str.upper().str.strip()

    # Detect which IDs belong to which dataset and swap if necessary
    # Check first few id1 values against both datasets
    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))

    sample_id1s = train_df["id1"].head(10).tolist()
    sample_id2s = train_df["id2"].head(10).tolist()

    id1_in_left = sum(1 for x in sample_id1s if x in left_ids)
    id1_in_right = sum(1 for x in sample_id1s if x in right_ids)
    id2_in_left = sum(1 for x in sample_id2s if x in left_ids)
    id2_in_right = sum(1 for x in sample_id2s if x in right_ids)

    # If id1 matches better with right and id2 matches better with left, swap columns
    if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
        train_df = train_df.rename(columns={"id1": "id2_temp", "id2": "id1"})
        train_df = train_df.rename(columns={"id2_temp": "id2"})

    # Convert labels to binary
    labels = (train_df["label"] == "TRUE").astype(int)

    # Create feature extractor
    feature_extractor = _create_feature_extractor(df_left, df_right, id_column)

    # Extract features
    pairs_df = train_df[["id1", "id2"]]
    features_df = feature_extractor.create_features(df_left, df_right, pairs_df, id_column, labels=labels)

    # Check if any pairs were skipped
    if len(features_df) < len(train_df):
        skipped = len(train_df) - len(features_df)
        print(f"    Note: {skipped} training pairs skipped (records not found in datasets)")

    # Prepare X and y
    id_cols = ["id1", "id2", "label"]
    feature_cols = [c for c in features_df.columns if c not in id_cols]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    # Create and train classifier
    if matcher_type == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif matcher_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif matcher_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif matcher_type == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")

    clf.fit(X, y)

    return clf, feature_extractor


def train_matcher_from_df(
    train_df: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    matcher_type: str,
    id_column: str = "id",
) -> tuple[Any, Any]:
    """Train a matcher using a training DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame with columns: id1, id2, label (TRUE/FALSE)
    df_left, df_right : pd.DataFrame
        Source dataframes for feature extraction
    matcher_type : str
        One of: "logreg", "rf", "gb", "xgb"
    id_column : str
        Name of ID column

    Returns
    -------
    tuple[classifier, feature_extractor]
        Trained classifier and feature extractor
    """
    train_df = train_df.copy()
    train_df["id1"] = train_df["id1"].astype(str)
    train_df["id2"] = train_df["id2"].astype(str)
    train_df["label"] = train_df["label"].astype(str).str.upper().str.strip()

    # Detect which IDs belong to which dataset and swap if necessary
    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))

    sample_id1s = train_df["id1"].head(10).tolist()
    sample_id2s = train_df["id2"].head(10).tolist()

    id1_in_left = sum(1 for x in sample_id1s if x in left_ids)
    id1_in_right = sum(1 for x in sample_id1s if x in right_ids)
    id2_in_left = sum(1 for x in sample_id2s if x in left_ids)
    id2_in_right = sum(1 for x in sample_id2s if x in right_ids)

    # If id1 matches better with right and id2 matches better with left, swap columns
    if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
        train_df = train_df.rename(columns={"id1": "id2_temp", "id2": "id1"})
        train_df = train_df.rename(columns={"id2_temp": "id2"})

    # Convert labels to binary
    labels = (train_df["label"] == "TRUE").astype(int)

    # Create feature extractor
    feature_extractor = _create_feature_extractor(df_left, df_right, id_column)

    # Extract features
    pairs_df = train_df[["id1", "id2"]]
    features_df = feature_extractor.create_features(df_left, df_right, pairs_df, id_column, labels=labels)

    # Check if any pairs were skipped
    if len(features_df) < len(train_df):
        skipped = len(train_df) - len(features_df)
        print(f"    Note: {skipped} training pairs skipped (records not found in datasets)")

    # Prepare X and y
    id_cols = ["id1", "id2", "label"]
    feature_cols = [c for c in features_df.columns if c not in id_cols]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    # Create and train classifier
    if matcher_type == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif matcher_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif matcher_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif matcher_type == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")

    clf.fit(X, y)

    return clf, feature_extractor


def evaluate_entity_matching_from_file(
    test_file: Path,
    matchers: dict[str, tuple[Any, Any]],  # {name: (classifier, feature_extractor)}
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str = "id",
) -> list[dict]:
    """Evaluate matchers on test file.

    Parameters
    ----------
    test_file : Path
        CSV with columns: id1, id2, label
    matchers : dict
        Dictionary mapping matcher name to (classifier, feature_extractor) tuple
    df_left, df_right : pd.DataFrame
        Source dataframes
    id_column : str
        Name of ID column

    Returns
    -------
    list[dict]
        List of results with F1, precision, recall per matcher
    """
    # Load test data - detect if file has header
    with open(test_file, 'r') as f:
        first_line = f.readline().strip()

    if 'id1' in first_line.lower() or 'label' in first_line.lower():
        test_df = pd.read_csv(test_file)
        test_df.columns = [c.lower() for c in test_df.columns]
    else:
        test_df = pd.read_csv(test_file, header=None, names=["id1", "id2", "label"])

    test_df["id1"] = test_df["id1"].astype(str)
    test_df["id2"] = test_df["id2"].astype(str)
    test_df["label"] = test_df["label"].astype(str).str.upper().str.strip()

    # Detect which IDs belong to which dataset and swap if necessary
    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))

    sample_id1s = test_df["id1"].head(10).tolist()
    sample_id2s = test_df["id2"].head(10).tolist()

    id1_in_left = sum(1 for x in sample_id1s if x in left_ids)
    id1_in_right = sum(1 for x in sample_id1s if x in right_ids)
    id2_in_left = sum(1 for x in sample_id2s if x in left_ids)
    id2_in_right = sum(1 for x in sample_id2s if x in right_ids)

    # If id1 matches better with right and id2 matches better with left, swap columns
    if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
        test_df = test_df.rename(columns={"id1": "id2_temp", "id2": "id1"})
        test_df = test_df.rename(columns={"id2_temp": "id2"})

    # Convert labels to binary (keep as series with index for alignment)
    test_df["label_binary"] = (test_df["label"] == "TRUE").astype(int)
    pairs_df = test_df[["id1", "id2"]]

    results = []

    for name, (clf, feature_extractor) in matchers.items():
        try:
            # Extract features using the matcher's feature extractor
            features_df = feature_extractor.create_features(df_left, df_right, pairs_df, id_column, labels=None)

            # Handle missing records: align test labels with extracted features
            # The feature extractor may skip pairs where records aren't found
            if len(features_df) != len(test_df):
                # Merge to get only the pairs that were successfully extracted
                features_df["id1"] = features_df["id1"].astype(str)
                features_df["id2"] = features_df["id2"].astype(str)
                merged = features_df.merge(
                    test_df[["id1", "id2", "label_binary"]],
                    on=["id1", "id2"],
                    how="inner"
                )
                y_true = merged["label_binary"].values
                # Get feature columns (exclude id and label columns)
                id_cols = ["id1", "id2", "label", "label_binary"]
                feature_cols = [c for c in merged.columns if c not in id_cols]
                X = merged[feature_cols].values
                skipped = len(test_df) - len(merged)
                if skipped > 0:
                    print(f"    Note: {skipped} test pairs skipped (records not found in datasets)")
            else:
                y_true = test_df["label_binary"].values
                # Prepare X
                id_cols = ["id1", "id2", "label"]
                feature_cols = [c for c in features_df.columns if c not in id_cols]
                X = features_df[feature_cols].values

            # Get predictions
            y_pred = clf.predict(X)

            # Calculate metrics
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            results.append({
                "matcher": name,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
            })
        except Exception as e:
            print(f"    Error evaluating {name}: {e}")
            results.append({
                "matcher": name,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "error": str(e),
            })

    return results


def _load_train_file_as_df(train_file: Path) -> pd.DataFrame:
    """Load a training file as a DataFrame with normalized columns.

    Parameters
    ----------
    train_file : Path
        CSV file with columns: id1, id2, label

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id1, id2, label (normalized)
    """
    # Detect if file has header
    with open(train_file, 'r') as f:
        first_line = f.readline().strip()

    if 'id1' in first_line.lower() or 'label' in first_line.lower():
        df = pd.read_csv(train_file)
        df.columns = [c.lower() for c in df.columns]
    else:
        df = pd.read_csv(train_file, header=None, names=["id1", "id2", "label"])

    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    return df


def _stratified_sample_to_match(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified sample source_df to match target_df's size and class distribution.

    Parameters
    ----------
    source_df : pd.DataFrame
        DataFrame to sample from (must have 'label' column)
    target_df : pd.DataFrame
        DataFrame whose size and class distribution to match (must have 'label' column)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Sampled source_df matching target_df's size and distribution
    """
    # Normalize labels
    source_df = source_df.copy()
    target_df = target_df.copy()
    source_df["label"] = source_df["label"].astype(str).str.upper().str.strip()
    target_df["label"] = target_df["label"].astype(str).str.upper().str.strip()

    # Get target distribution
    target_positives = (target_df["label"] == "TRUE").sum()
    target_negatives = (target_df["label"] == "FALSE").sum()
    target_total = len(target_df)

    # Get source counts
    source_positives_df = source_df[source_df["label"] == "TRUE"]
    source_negatives_df = source_df[source_df["label"] == "FALSE"]

    # Sample to match target distribution
    # If source doesn't have enough of a class, take all available
    n_pos_sample = min(target_positives, len(source_positives_df))
    n_neg_sample = min(target_negatives, len(source_negatives_df))

    sampled_pos = source_positives_df.sample(n=n_pos_sample, random_state=random_state)
    sampled_neg = source_negatives_df.sample(n=n_neg_sample, random_state=random_state)

    result = pd.concat([sampled_pos, sampled_neg], ignore_index=True)

    # Shuffle the result
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return result


def _sample_with_ratio(
    df: pd.DataFrame,
    pos_ratio: int = 1,
    neg_ratio: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample a DataFrame to achieve a specific positive:negative ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'label' column (TRUE/FALSE)
    pos_ratio : int
        Ratio part for positives (default 1)
    neg_ratio : int
        Ratio part for negatives (default 2)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame with pos_ratio:neg_ratio distribution
    """
    df = df.copy()
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    positives = df[df["label"] == "TRUE"]
    negatives = df[df["label"] == "FALSE"]

    n_pos = len(positives)
    n_neg = len(negatives)

    # Calculate target counts based on ratio
    # Use all positives, sample negatives to match ratio
    # Or use all negatives, sample positives if negatives are limiting
    target_neg_for_all_pos = n_pos * neg_ratio // pos_ratio
    target_pos_for_all_neg = n_neg * pos_ratio // neg_ratio

    if target_neg_for_all_pos <= n_neg:
        # We have enough negatives, use all positives
        sampled_pos = positives
        sampled_neg = negatives.sample(n=target_neg_for_all_pos, random_state=random_state)
    else:
        # Not enough negatives, use all negatives and sample positives
        sampled_neg = negatives
        sampled_pos = positives.sample(n=min(target_pos_for_all_neg, n_pos), random_state=random_state)

    result = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return result


def _evaluate_single_matcher(
    test_file: Path,
    classifier,
    feature_extractor,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str = "id",
) -> dict:
    """Evaluate a single classifier on test file and return metrics.

    Parameters
    ----------
    test_file : Path
        CSV with columns: id1, id2, label
    classifier : sklearn classifier
        Trained classifier with predict() method
    feature_extractor : FeatureExtractor
        Feature extractor to use for test pairs
    df_left, df_right : pd.DataFrame
        Source dataframes
    id_column : str
        Name of ID column

    Returns
    -------
    dict
        Dictionary with f1, precision, recall metrics
    """
    # Load test data - detect if file has header
    with open(test_file, 'r') as f:
        first_line = f.readline().strip()

    if 'id1' in first_line.lower() or 'label' in first_line.lower():
        test_df = pd.read_csv(test_file)
        test_df.columns = [c.lower() for c in test_df.columns]
    else:
        test_df = pd.read_csv(test_file, header=None, names=["id1", "id2", "label"])

    test_df["id1"] = test_df["id1"].astype(str)
    test_df["id2"] = test_df["id2"].astype(str)
    test_df["label"] = test_df["label"].astype(str).str.upper().str.strip()

    # Detect which IDs belong to which dataset and swap if necessary
    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))

    sample_id1s = test_df["id1"].head(10).tolist()
    sample_id2s = test_df["id2"].head(10).tolist()

    id1_in_left = sum(1 for x in sample_id1s if x in left_ids)
    id1_in_right = sum(1 for x in sample_id1s if x in right_ids)
    id2_in_left = sum(1 for x in sample_id2s if x in left_ids)
    id2_in_right = sum(1 for x in sample_id2s if x in right_ids)

    # If id1 matches better with right and id2 matches better with left, swap columns
    if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
        test_df = test_df.rename(columns={"id1": "id2_temp", "id2": "id1"})
        test_df = test_df.rename(columns={"id2_temp": "id2"})

    # Extract features
    pairs_df = test_df[["id1", "id2"]]
    features_df = feature_extractor.create_features(df_left, df_right, pairs_df, id_column, labels=None)

    # Handle missing records: align test labels with extracted features
    if len(features_df) != len(test_df):
        features_df["id1"] = features_df["id1"].astype(str)
        features_df["id2"] = features_df["id2"].astype(str)
        test_df["label_binary"] = (test_df["label"] == "TRUE").astype(int)
        merged = features_df.merge(
            test_df[["id1", "id2", "label_binary"]],
            on=["id1", "id2"],
            how="inner"
        )
        y_true = merged["label_binary"].values
        id_cols = ["id1", "id2", "label", "label_binary"]
        feature_cols = [c for c in merged.columns if c not in id_cols]
        X = merged[feature_cols].values
    else:
        y_true = (test_df["label"] == "TRUE").astype(int).values
        id_cols = ["id1", "id2", "label"]
        feature_cols = [c for c in features_df.columns if c not in id_cols]
        X = features_df[feature_cols].values

    # Get predictions
    y_pred = classifier.predict(X)

    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def _format_record(
    df: pd.DataFrame,
    record_id: str,
    id_column: str = "id",
    max_cols: int = 4,
) -> str:
    """Format a record's key attributes for display.

    Args:
        df: DataFrame containing the record
        record_id: ID of the record to format
        id_column: Name of ID column
        max_cols: Maximum number of columns to show

    Returns:
        Formatted string like "name=Foo, year=2020"
    """
    row = df[df[id_column].astype(str) == str(record_id)]
    if row.empty:
        return "(not found)"

    row = row.iloc[0]
    # Prefer these columns if they exist
    priority_cols = ["name", "title", "releaseYear", "year", "developer", "platform", "publisher"]
    cols_to_show = []

    for col in priority_cols:
        if col in df.columns and col != id_column and pd.notna(row.get(col)):
            cols_to_show.append(col)
            if len(cols_to_show) >= max_cols:
                break

    # Fill remaining slots with other columns
    if len(cols_to_show) < max_cols:
        for col in df.columns:
            if col not in cols_to_show and col != id_column and pd.notna(row.get(col)):
                cols_to_show.append(col)
                if len(cols_to_show) >= max_cols:
                    break

    if not cols_to_show:
        return "(no attributes)"

    parts = []
    for col in cols_to_show:
        val = row.get(col)
        # Truncate long values
        val_str = str(val)
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        parts.append(f"{col}={val_str}")

    return ", ".join(parts)


def load_xml_fusion_test_set(test_dir: Path) -> list[dict] | None:
    """Load XML fusion test set containing gold standard fused entities.

    The XML test set contains fused entity records with their attributes,
    representing the expected output of the fusion process.

    Args:
        test_dir: Directory containing test_set.xml

    Returns:
        List of dicts with 'id' and other attributes, or None if not found
    """
    import xml.etree.ElementTree as ET

    xml_path = test_dir / "test_set.xml"
    if not xml_path.exists():
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        entities = []
        # Handle various root element names (videogames, companies, etc.)
        for child in root:
            entity = {}
            for elem in child:
                if elem.text and elem.text.strip():
                    entity[elem.tag] = elem.text.strip()
                elif len(elem) > 0:
                    # Handle nested elements (like genres)
                    entity[elem.tag] = [sub.text for sub in elem if sub.text]
            if entity.get("id"):
                entities.append(entity)

        return entities if entities else None
    except Exception as e:
        print(f"  Failed to parse XML test set: {e}")
        return None


def evaluate_fusion_coverage(
    xml_test_entities: list[dict],
    fused_df: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate fusion coverage against XML test set.

    Checks if test set entity IDs appear in the fused output.

    Args:
        xml_test_entities: List of gold standard entities from XML
        fused_df: Fused dataset with _fusion_sources column

    Returns:
        Dictionary with coverage metrics
    """
    import ast

    # Build reverse lookup from source ID to fused row
    source_to_row: dict[str, int] = {}
    for idx, row in fused_df.iterrows():
        sources_str = row.get("_fusion_sources", "")
        if pd.isna(sources_str) or not sources_str:
            # Try _id column as fallback
            if "_id" in fused_df.columns and pd.notna(row.get("_id")):
                source_to_row[str(row["_id"])] = idx
            continue
        try:
            sources = ast.literal_eval(sources_str)
            for src_id in sources:
                source_to_row[str(src_id)] = idx
        except (ValueError, SyntaxError):
            if "_id" in fused_df.columns and pd.notna(row.get("_id")):
                source_to_row[str(row["_id"])] = idx

    # Check coverage of test entities
    found = 0
    not_found = 0
    not_found_ids = []

    for entity in xml_test_entities:
        test_id = str(entity.get("id", ""))
        if test_id in source_to_row:
            found += 1
        else:
            not_found += 1
            not_found_ids.append(test_id)

    total = len(xml_test_entities)
    coverage = found / total if total > 0 else 0.0

    return {
        "total_test_entities": total,
        "found_in_fusion": found,
        "not_found": not_found,
        "coverage": coverage,
        "not_found_ids": not_found_ids[:10],  # First 10 for display
    }


def load_test_sets(
    test_dir: Path,
    dataset_names: list[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load held-out test sets from a directory.

    Test files should be named: {left}_2_{right}_test.csv
    Format: id1,id2,label (no header, or header with those column names)

    Args:
        test_dir: Directory containing test CSV files
        dataset_names: List of dataset names to look for

    Returns:
        Dictionary mapping (left_name, right_name) to test DataFrame
    """
    test_sets = {}
    if not test_dir or not test_dir.exists():
        return test_sets

    # Find all test files matching the pattern
    for test_file in test_dir.glob("*_2_*_test.csv"):
        filename = test_file.stem  # e.g., "metacritic_2_dbpedia_test"
        if not filename.endswith("_test"):
            continue

        # Parse the filename: {left}_2_{right}_test
        name_part = filename[:-5]  # Remove "_test"
        if "_2_" not in name_part:
            continue

        parts = name_part.split("_2_", 1)
        if len(parts) != 2:
            continue

        left_name, right_name = parts

        # Check if both datasets are in our pipeline
        if left_name not in dataset_names or right_name not in dataset_names:
            print(f"  Skipping {test_file.name}: datasets not in pipeline")
            continue

        # Load the test set (try with and without header)
        try:
            # First try reading with header detection
            df = pd.read_csv(test_file, header=None, names=["id1", "id2", "label"])
            # Check if first row looks like a header
            if df.iloc[0]["id1"] == "id1" or df.iloc[0]["label"] in ["label", "TRUE", "FALSE"]:
                if df.iloc[0]["label"] not in ["TRUE", "FALSE"]:
                    df = df.iloc[1:].reset_index(drop=True)

            # Normalize label column
            df["label"] = df["label"].astype(str).str.upper().str.strip()

            test_sets[(left_name, right_name)] = df
            n_pos = (df["label"] == "TRUE").sum()
            n_neg = (df["label"] == "FALSE").sum()
            print(f"  Loaded {test_file.name}: {len(df)} pairs ({n_pos} positive, {n_neg} negative)")
        except Exception as e:
            print(f"  Failed to load {test_file.name}: {e}")

    return test_sets


def evaluate_blocking_coverage(
    test_set: pd.DataFrame,
    blocker_spec: dict,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    id_column: str = "id",
) -> dict[str, Any]:
    """Evaluate how many test pairs are covered by the blocker.

    Args:
        test_set: Test set DataFrame with id1, id2, label columns
        blocker_spec: Blocker specification dict
        df_left: Left dataset
        df_right: Right dataset
        left_name: Name of left dataset
        right_name: Name of right dataset
        id_column: ID column name

    Returns:
        Dictionary with blocking coverage metrics
    """
    from PyDI.entitymatching.blocking import blocker_from_spec

    print(f"\n  === Blocking Coverage Analysis ===")
    print(f"  Blocker spec: {blocker_spec}")

    # Determine correct dataset assignment based on ID prefixes in test set
    test_set = test_set.copy()
    test_set["id1"] = test_set["id1"].astype(str)
    test_set["id2"] = test_set["id2"].astype(str)

    test_sample = test_set.iloc[0]
    test_id1_prefix = str(test_sample["id1"]).split("_")[0]

    # Match datasets to test file ordering
    if test_id1_prefix == left_name:
        blocker_left, blocker_right = df_left, df_right
    else:
        blocker_left, blocker_right = df_right, df_left

    blocker = blocker_from_spec(blocker_spec, blocker_left, blocker_right, id_column)

    # Collect all candidate pairs from blocker
    candidate_pairs = set()
    for batch in blocker:
        if batch is not None and not batch.empty:
            for _, row in batch.iterrows():
                candidate_pairs.add((str(row["id1"]), str(row["id2"])))
                candidate_pairs.add((str(row["id2"]), str(row["id1"])))  # Both orderings

    print(f"  Total candidate pairs from blocker: {len(candidate_pairs) // 2}")

    # Check coverage of positive test pairs
    positive_pairs = test_set[test_set["label"] == "TRUE"]
    blocked_positives = 0
    missed_by_blocker = []

    for _, row in positive_pairs.iterrows():
        pair = (str(row["id1"]), str(row["id2"]))
        pair_rev = (str(row["id2"]), str(row["id1"]))
        if pair in candidate_pairs or pair_rev in candidate_pairs:
            blocked_positives += 1
        else:
            missed_by_blocker.append({"id1": row["id1"], "id2": row["id2"]})

    blocking_recall = blocked_positives / len(positive_pairs) if len(positive_pairs) > 0 else 0.0
    print(f"  Positive test pairs: {len(positive_pairs)}")
    print(f"  Covered by blocker: {blocked_positives} ({blocking_recall:.1%})")
    print(f"  Missed by blocker: {len(positive_pairs) - blocked_positives} ({1 - blocking_recall:.1%})")

    result = {
        "total_candidates": len(candidate_pairs) // 2,
        "positive_pairs": len(positive_pairs),
        "covered_by_blocker": blocked_positives,
        "blocking_recall": blocking_recall,
        "missed_by_blocker": missed_by_blocker,
    }

    if missed_by_blocker:
        print(f"\n  --- Sample Pairs Missed by Blocker ---")
        for ex in missed_by_blocker[:5]:
            print(f"    {ex['id1']} <-> {ex['id2']}")

    return result


def evaluate_matching(
    test_set: pd.DataFrame,
    correspondences: pd.DataFrame,
    blocking_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate matching performance against test set.

    Args:
        test_set: Test set DataFrame with id1, id2, label columns
        correspondences: Correspondences DataFrame with id1, id2, score columns
        blocking_coverage: Optional blocking coverage result for error attribution

    Returns:
        Dictionary with matching metrics and error examples
    """
    test_set = test_set.copy()
    test_set["id1"] = test_set["id1"].astype(str)
    test_set["id2"] = test_set["id2"].astype(str)

    correspondences = correspondences.copy()
    correspondences["id1"] = correspondences["id1"].astype(str)
    correspondences["id2"] = correspondences["id2"].astype(str)

    # Create sets of predicted matches (both orderings for lookup)
    predicted_matches = set(zip(correspondences["id1"], correspondences["id2"]))
    predicted_matches_reversed = set(zip(correspondences["id2"], correspondences["id1"]))

    # Get pairs missed by blocker for attribution
    missed_by_blocker_set = set()
    if blocking_coverage and blocking_coverage.get("missed_by_blocker"):
        for ex in blocking_coverage["missed_by_blocker"]:
            missed_by_blocker_set.add((str(ex["id1"]), str(ex["id2"])))
            missed_by_blocker_set.add((str(ex["id2"]), str(ex["id1"])))

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    false_negative_examples: list[dict] = []
    false_positive_examples: list[dict] = []

    for _, row in test_set.iterrows():
        pair = (str(row["id1"]), str(row["id2"]))
        pair_reversed = (str(row["id2"]), str(row["id1"]))
        is_match = row["label"] == "TRUE"

        # Check both orderings
        is_predicted = (
            pair in predicted_matches
            or pair_reversed in predicted_matches
            or pair in predicted_matches_reversed
            or pair_reversed in predicted_matches_reversed
        )

        if is_match and is_predicted:
            true_positives += 1
        elif is_match and not is_predicted:
            false_negatives += 1
            if len(false_negative_examples) < 20:
                # Attribute error to blocker or matcher
                missed_by_blocker = pair in missed_by_blocker_set or pair_reversed in missed_by_blocker_set
                false_negative_examples.append({
                    "id1": row["id1"],
                    "id2": row["id2"],
                    "label": row["label"],
                    "error_type": "false_negative",
                    "missed_by": "blocker" if missed_by_blocker else "matcher",
                })
        elif not is_match and is_predicted:
            false_positives += 1
            if len(false_positive_examples) < 20:
                false_positive_examples.append({
                    "id1": row["id1"],
                    "id2": row["id2"],
                    "label": row["label"],
                    "error_type": "false_positive",
                    "missed_by": "matcher",
                })
        else:
            true_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / len(test_set) if len(test_set) > 0 else 0.0

    return {
        "test_pairs": len(test_set),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_negative_examples": false_negative_examples,
        "false_positive_examples": false_positive_examples,
    }


def save_error_examples(
    error_examples: list[dict],
    df_left: pd.DataFrame | None,
    df_right: pd.DataFrame | None,
    output_path: Path,
    left_name: str,
    right_name: str,
    id_column: str = "id",
) -> None:
    """Save error examples with full record details.

    Args:
        error_examples: List of error example dicts
        df_left: Left dataset for record lookup
        df_right: Right dataset for record lookup
        output_path: Path to save CSV
        left_name: Name of left dataset (for ID matching)
        right_name: Name of right dataset (for ID matching)
        id_column: ID column name
    """
    if not error_examples:
        return

    error_rows = []
    for ex in error_examples:
        error_row = {
            "id1": ex["id1"],
            "id2": ex["id2"],
            "label": ex["label"],
            "error_type": ex["error_type"],
            "missed_by": ex.get("missed_by", "unknown"),
        }

        id1_str = str(ex["id1"])
        id2_str = str(ex["id2"])

        # Determine which ID belongs to which dataset based on prefix
        id1_prefix = id1_str.split("_")[0]
        id2_prefix = id2_str.split("_")[0]

        # Match IDs to correct datasets
        if df_left is not None:
            left_id = id1_str if id1_prefix == left_name else id2_str
            left_row = df_left[df_left[id_column].astype(str) == left_id]
            if not left_row.empty:
                for col in df_left.columns:
                    if col != id_column:
                        error_row[f"left_{col}"] = left_row.iloc[0].get(col)

        if df_right is not None:
            right_id = id2_str if id2_prefix == right_name else id1_str
            right_row = df_right[df_right[id_column].astype(str) == right_id]
            if not right_row.empty:
                for col in df_right.columns:
                    if col != id_column:
                        error_row[f"right_{col}"] = right_row.iloc[0].get(col)

        error_rows.append(error_row)

    if error_rows:
        error_df = pd.DataFrame(error_rows)
        error_df.to_csv(output_path, index=False)
        print(f"  Saved {len(error_rows)} error examples to: {output_path}")


def evaluate_fusion(
    test_sets: dict[tuple[str, str], pd.DataFrame],
    fused_df: pd.DataFrame,
    id_column: str = "id",
) -> list[dict]:
    """Evaluate fusion quality against test sets.

    Checks if positive pairs ended up in the same fused row.

    Args:
        test_sets: Dictionary of test sets
        fused_df: Fused dataset
        id_column: ID column name

    Returns:
        List of fusion evaluation results per dataset pair
    """
    import ast

    fusion_eval_results = []

    # Build reverse lookup from source ID to fused row index
    # _fusion_sources contains string like "['id1', 'id2']"
    source_to_row: dict[str, int] = {}

    for idx, row in fused_df.iterrows():
        sources_str = row.get("_fusion_sources", "")
        if pd.isna(sources_str) or not sources_str:
            continue
        try:
            # Parse the string representation of the list
            sources = ast.literal_eval(sources_str)
            for src_id in sources:
                source_to_row[str(src_id)] = idx
        except (ValueError, SyntaxError):
            # If parsing fails, try to use _id column as fallback
            if "_id" in fused_df.columns and pd.notna(row["_id"]):
                source_to_row[str(row["_id"])] = idx

    for (left_name, right_name), test_set in test_sets.items():
        positive_pairs = test_set[test_set["label"] == "TRUE"]

        correct_fusions = 0
        incorrect_fusions = 0
        not_found = 0

        for _, row in positive_pairs.iterrows():
            id1, id2 = str(row["id1"]), str(row["id2"])

            # Find which fused row each ID belongs to
            row_idx_1 = source_to_row.get(id1)
            row_idx_2 = source_to_row.get(id2)

            if row_idx_1 is None or row_idx_2 is None:
                not_found += 1
            elif row_idx_1 == row_idx_2:
                correct_fusions += 1
            else:
                incorrect_fusions += 1

        total = len(positive_pairs)
        fusion_accuracy = correct_fusions / total if total > 0 else 0.0

        fusion_eval_results.append({
            "left": left_name,
            "right": right_name,
            "positive_pairs": total,
            "correctly_fused": correct_fusions,
            "incorrectly_separated": incorrect_fusions,
            "not_found": not_found,
            "fusion_accuracy": fusion_accuracy,
        })

        print(f"  {left_name} <-> {right_name}:")
        print(f"    Positive test pairs: {total}")
        print(f"    Correctly fused: {correct_fusions} ({fusion_accuracy:.1%})")
        print(f"    Incorrectly separated: {incorrect_fusions}")
        print(f"    Not found in fused data: {not_found}")

    return fusion_eval_results


def run_test_evaluation(
    test_dir: Path,
    output_dir: Path,
    results: dict[str, pd.DataFrame],
    blocking_choices: dict[tuple[str, str], dict],
    matching_dir: Path,
    fusion_dir: Path,
    id_column: str = "id",
) -> tuple[list[dict], list[dict]]:
    """Run full test set evaluation.

    Args:
        test_dir: Directory containing test CSV files
        output_dir: Directory for output files
        results: Dictionary of normalized datasets
        blocking_choices: Dictionary of blocker specs per pair
        matching_dir: Directory containing correspondence files
        fusion_dir: Directory containing fusion results
        id_column: ID column name

    Returns:
        Tuple of (test_evaluation_results, fusion_evaluation_results)
    """
    print(f"\n=== Step 7: Test Set Evaluation ===")
    print(f"Test directory: {test_dir}")

    dataset_names = list(results.keys())
    test_sets = load_test_sets(test_dir, dataset_names)

    test_output_dir = output_dir / "entity_resolution" / "test_evaluation"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    test_evaluation_results = []
    fusion_eval_results = []

    # Check for XML fusion test set early (will be used later)
    xml_test_entities = load_xml_fusion_test_set(test_dir)

    if not test_sets:
        print("  No matching CSV test sets found for entity matching evaluation.")
        # Still continue to check for XML fusion test set below

    for (left_name, right_name), test_set in test_sets.items():
        print(f"\n--- Evaluating {left_name} <-> {right_name} ---")

        # Get datasets
        df_left = results.get(left_name)
        df_right = results.get(right_name)

        # Evaluate blocking coverage
        blocker_spec = blocking_choices.get((left_name, right_name)) or blocking_choices.get((right_name, left_name))
        blocking_coverage = None

        if blocker_spec and df_left is not None and df_right is not None:
            try:
                blocking_coverage = evaluate_blocking_coverage(
                    test_set=test_set,
                    blocker_spec=blocker_spec,
                    df_left=df_left,
                    df_right=df_right,
                    left_name=left_name,
                    right_name=right_name,
                    id_column=id_column,
                )
            except Exception as e:
                print(f"  Blocking coverage analysis failed: {e}")
                import traceback
                traceback.print_exc()

        # Load correspondences
        corr_path = matching_dir / f"correspondences_{left_name}_{right_name}.csv"
        corr_path_alt = matching_dir / f"correspondences_{right_name}_{left_name}.csv"

        if corr_path.exists():
            print(f"\n  === Matcher Evaluation ===")
            print(f"  Loading correspondences from: {corr_path.name}")
        elif corr_path_alt.exists():
            corr_path = corr_path_alt
            print(f"\n  === Matcher Evaluation ===")
            print(f"  Loading correspondences from: {corr_path.name} (swapped order)")
        else:
            print(f"  No correspondences file found at {corr_path} or {corr_path_alt}")
            continue

        correspondences = pd.read_csv(corr_path)
        print(f"  Correspondences loaded: {len(correspondences)} predicted matches")

        # Evaluate matching
        match_result = evaluate_matching(
            test_set=test_set,
            correspondences=correspondences,
            blocking_coverage=blocking_coverage,
        )

        # Add pair names to result
        match_result["left"] = left_name
        match_result["right"] = right_name

        # Add blocking coverage to result
        if blocking_coverage:
            match_result["blocking_recall"] = blocking_coverage["blocking_recall"]
            match_result["blocked_candidates"] = blocking_coverage["total_candidates"]

        test_evaluation_results.append(match_result)

        print(f"  Test pairs: {match_result['test_pairs']}")
        print(f"  TP: {match_result['true_positives']}, FP: {match_result['false_positives']}, "
              f"FN: {match_result['false_negatives']}, TN: {match_result['true_negatives']}")
        print(f"  Precision: {match_result['precision']:.3f}, Recall: {match_result['recall']:.3f}, "
              f"F1: {match_result['f1']:.3f}, Accuracy: {match_result['accuracy']:.3f}")

        # Print error attribution
        fn_examples = match_result.get("false_negative_examples", [])
        if fn_examples:
            blocker_errors = sum(1 for ex in fn_examples if ex.get("missed_by") == "blocker")
            matcher_errors = sum(1 for ex in fn_examples if ex.get("missed_by") == "matcher")
            total_fn = match_result["false_negatives"]

            if blocking_coverage:
                total_missed_blocker = len(blocking_coverage.get("missed_by_blocker", []))
                print(f"\n  Error Attribution:")
                print(f"    False negatives from blocker: ~{total_missed_blocker} (blocker recall: {blocking_coverage['blocking_recall']:.1%})")
                print(f"    False negatives from matcher: ~{total_fn - total_missed_blocker}")

        # Save error examples
        all_errors = match_result.get("false_negative_examples", []) + match_result.get("false_positive_examples", [])
        if all_errors:
            error_path = test_output_dir / f"errors_{left_name}_{right_name}.csv"
            save_error_examples(
                error_examples=all_errors,
                df_left=df_left,
                df_right=df_right,
                output_path=error_path,
                left_name=left_name,
                right_name=right_name,
                id_column=id_column,
            )

            # Print sample errors with record details
            fn_examples = match_result.get("false_negative_examples", [])
            fp_examples = match_result.get("false_positive_examples", [])

            if fn_examples:
                print(f"\n  --- Sample False Negatives (missed matches) ---")
                for ex in fn_examples[:5]:
                    missed_by = f" [missed by {ex.get('missed_by', '?')}]" if ex.get("missed_by") else ""
                    id1_str, id2_str = str(ex["id1"]), str(ex["id2"])

                    # Determine which ID belongs to which dataset
                    id1_prefix = id1_str.split("_")[0]
                    if df_left is not None and df_right is not None:
                        if id1_prefix == left_name:
                            left_attrs = _format_record(df_left, id1_str, id_column)
                            right_attrs = _format_record(df_right, id2_str, id_column)
                        else:
                            left_attrs = _format_record(df_left, id2_str, id_column)
                            right_attrs = _format_record(df_right, id1_str, id_column)
                        print(f"    {id1_str} <-> {id2_str}{missed_by}")
                        print(f"      Left:  {left_attrs}")
                        print(f"      Right: {right_attrs}")
                    else:
                        print(f"    {id1_str} <-> {id2_str}{missed_by}")

            if fp_examples:
                print(f"\n  --- Sample False Positives (incorrect matches) ---")
                for ex in fp_examples[:5]:
                    id1_str, id2_str = str(ex["id1"]), str(ex["id2"])

                    # Determine which ID belongs to which dataset
                    id1_prefix = id1_str.split("_")[0]
                    if df_left is not None and df_right is not None:
                        if id1_prefix == left_name:
                            left_attrs = _format_record(df_left, id1_str, id_column)
                            right_attrs = _format_record(df_right, id2_str, id_column)
                        else:
                            left_attrs = _format_record(df_left, id2_str, id_column)
                            right_attrs = _format_record(df_right, id1_str, id_column)
                        print(f"    {id1_str} <-> {id2_str}")
                        print(f"      Left:  {left_attrs}")
                        print(f"      Right: {right_attrs}")
                    else:
                        print(f"    {id1_str} <-> {id2_str}")

    # Save test evaluation results
    if test_evaluation_results:
        # Remove example lists for CSV output
        csv_results = []
        for r in test_evaluation_results:
            csv_row = {k: v for k, v in r.items() if not k.endswith("_examples")}
            csv_results.append(csv_row)

        test_eval_df = pd.DataFrame(csv_results)
        test_eval_path = test_output_dir / "test_evaluation_results.csv"
        test_eval_df.to_csv(test_eval_path, index=False)
        print(f"\nSaved test evaluation results to: {test_eval_path}")

        print("\n--- Test Set Evaluation Summary ---")
        print(test_eval_df.to_string(index=False))

    # Evaluate fusion
    # Use fused_clean.csv instead of fused.csv because the _fusion_metadata column
    # in fused.csv contains unquoted commas that break CSV parsing
    fused_path = fusion_dir / "fused_clean.csv"
    if not fused_path.exists():
        fused_path = fusion_dir / "fused.csv"  # Fallback to original

    print(f"\n  Fusion file path: {fused_path.absolute()}")
    print(f"  Fusion file exists: {fused_path.exists()}")

    # Try XML test set for fusion coverage evaluation (xml_test_entities loaded earlier)
    if xml_test_entities and fused_path.exists():
        print("\n--- Fusion Coverage Evaluation (XML Test Set) ---")
        print(f"  Loaded {len(xml_test_entities)} gold standard entities from test_set.xml")

        fused_df = pd.read_csv(fused_path)
        print(f"  Fused dataframe: {len(fused_df)} rows, columns: {fused_df.columns.tolist()}")
        coverage_result = evaluate_fusion_coverage(xml_test_entities, fused_df)

        print(f"  Total test entities: {coverage_result['total_test_entities']}")
        print(f"  Found in fusion output: {coverage_result['found_in_fusion']} ({coverage_result['coverage']:.1%})")
        print(f"  Not found: {coverage_result['not_found']}")

        if coverage_result['not_found_ids']:
            print(f"  Sample missing IDs: {coverage_result['not_found_ids'][:5]}")

        # Save coverage result
        coverage_path = test_output_dir / "fusion_coverage_results.csv"
        pd.DataFrame([{
            "total_test_entities": coverage_result["total_test_entities"],
            "found_in_fusion": coverage_result["found_in_fusion"],
            "not_found": coverage_result["not_found"],
            "coverage": coverage_result["coverage"],
        }]).to_csv(coverage_path, index=False)
        print(f"\nSaved fusion coverage results to: {coverage_path}")

    # Also run pair-based fusion evaluation if CSV test sets exist
    if fused_path.exists() and test_sets:
        print("\n--- Fusion Quality Evaluation (CSV Test Sets) ---")
        fused_df = pd.read_csv(fused_path)

        if "_fusion_source_datasets" in fused_df.columns:
            fusion_eval_results = evaluate_fusion(test_sets, fused_df, id_column)

            if fusion_eval_results:
                fusion_eval_df = pd.DataFrame(fusion_eval_results)
                fusion_eval_path = test_output_dir / "fusion_evaluation_results.csv"
                fusion_eval_df.to_csv(fusion_eval_path, index=False)
                print(f"\nSaved fusion evaluation results to: {fusion_eval_path}")
        else:
            print("  Fused dataset lacks source tracking metadata; skipping pair-based fusion evaluation.")

    # Training Data Quality Comparison
    # Compare auto-generated (pipeline) training data vs manually curated training data
    entitymatching_dir = test_dir.parent / "entitymatching"
    if entitymatching_dir.exists():
        print("\n--- Training Data Quality Comparison ---")

        training_comparison_results = []
        manual_train_files = list(entitymatching_dir.glob("*_train.csv"))

        if not manual_train_files:
            print("  No manual train files found (*_train.csv). Skipping training data comparison.")
        else:
            print(f"  Found {len(manual_train_files)} manual train files")

        for manual_train_file in manual_train_files:
            try:
                # Parse left_2_right from filename
                parts = manual_train_file.stem.replace("_train", "").split("_2_")
                if len(parts) != 2:
                    print(f"  Skipping {manual_train_file.name}: could not parse left_2_right from filename")
                    continue
                left_name, right_name = parts
                datasets_str = f"{left_name}_2_{right_name}"

                test_file = entitymatching_dir / f"{left_name}_2_{right_name}_test.csv"
                if not test_file.exists():
                    # Try swapped order
                    test_file = entitymatching_dir / f"{right_name}_2_{left_name}_test.csv"
                if not test_file.exists():
                    print(f"  Skipping {datasets_str}: no test file found")
                    continue

                # Get datasets
                df_left = results.get(left_name)
                df_right = results.get(right_name)

                if df_left is None or df_right is None:
                    available = list(results.keys())
                    print(f"  Skipping {datasets_str}: dataset not found in results (available: {available})")
                    continue

                # Find pipeline (auto) training files - both hybrid and faiss
                training_dir = output_dir / "entity_resolution" / "training"

                # Helper to find training file with various naming patterns
                def find_training_file(prefix: str) -> Path:
                    """Find training file with given prefix, trying both dataset orderings."""
                    patterns = [
                        training_dir / f"similarity_training_{prefix}_{left_name}_{right_name}.csv",
                        training_dir / f"similarity_training_{prefix}_{right_name}_{left_name}.csv",
                    ]
                    for p in patterns:
                        if p.exists():
                            return p
                    return patterns[0]  # Return first pattern (will show as NOT FOUND)

                # FAISS training files (with faiss_ prefix)
                faiss_train_file = find_training_file("faiss")

                # Also check for old naming convention (no prefix)
                faiss_old_path = training_dir / f"similarity_training_{left_name}_{right_name}.csv"
                faiss_old_alt = training_dir / f"similarity_training_{right_name}_{left_name}.csv"
                if not faiss_train_file.exists():
                    if faiss_old_path.exists():
                        faiss_train_file = faiss_old_path
                    elif faiss_old_alt.exists():
                        faiss_train_file = faiss_old_alt

                # Hybrid training files (with hybrid_ prefix)
                hybrid_train_file = find_training_file("hybrid")

                # Augmented training files (from active learning)
                augmented_path = training_dir / f"training_{left_name}_{right_name}_augmented.csv"
                augmented_alt = training_dir / f"training_{right_name}_{left_name}_augmented.csv"
                augmented_train_file = augmented_path if augmented_path.exists() else augmented_alt

                print(f"\n  {datasets_str}:")
                print(f"    Manual train file: {manual_train_file.name}")
                print(f"    Auto (faiss) file: {faiss_train_file.name if faiss_train_file.exists() else 'NOT FOUND'}")
                print(f"    Auto (hybrid) file: {hybrid_train_file.name if hybrid_train_file.exists() else 'NOT FOUND'}")
                print(f"    Auto (augmented) file: {augmented_train_file.name if augmented_train_file.exists() else 'NOT FOUND'}")
                print(f"    Test file: {test_file.name}")

                # Load training files as DataFrames
                manual_df = _load_train_file_as_df(manual_train_file)
                faiss_df = _load_train_file_as_df(faiss_train_file) if faiss_train_file.exists() else None
                hybrid_df = _load_train_file_as_df(hybrid_train_file) if hybrid_train_file.exists() else None
                augmented_df = _load_train_file_as_df(augmented_train_file) if augmented_train_file.exists() else None

                # Collect auto datasets for comparison
                auto_datasets = []
                if faiss_df is not None:
                    auto_datasets.append(("auto (faiss)", faiss_df))
                if hybrid_df is not None:
                    auto_datasets.append(("auto (hybrid)", hybrid_df))
                if augmented_df is not None:
                    auto_datasets.append(("auto (augmented)", augmented_df))

                manual_pos = (manual_df["label"] == "TRUE").sum()
                manual_neg = (manual_df["label"] == "FALSE").sum()
                print(f"    Manual set (raw): {len(manual_df)} pairs ({manual_pos} pos, {manual_neg} neg)")

                for auto_name, auto_df in auto_datasets:
                    auto_pos = (auto_df["label"] == "TRUE").sum()
                    auto_neg = (auto_df["label"] == "FALSE").sum()
                    print(f"    {auto_name} (raw): {len(auto_df)} pairs ({auto_pos} pos, {auto_neg} neg)")

                # Sample auto datasets to 1:2 ratio first
                auto_datasets_sampled = []
                for auto_name, auto_df in auto_datasets:
                    auto_sampled = _sample_with_ratio(auto_df, pos_ratio=1, neg_ratio=2, random_state=42)
                    auto_datasets_sampled.append((auto_name, auto_sampled))
                    s_pos = (auto_sampled["label"] == "TRUE").sum()
                    s_neg = (auto_sampled["label"] == "FALSE").sum()
                    print(f"    {auto_name} (1:2 sampled): {len(auto_sampled)} pairs ({s_pos} pos, {s_neg} neg)")

                # Find smallest non-augmented auto dataset size to cap manual
                non_augmented_sizes = [
                    len(df) for name, df in auto_datasets_sampled
                    if "augmented" not in name.lower()
                ]
                max_manual_size = min(non_augmented_sizes) if non_augmented_sizes else None

                # Sample manual to 1:2 ratio, capped to smallest non-augmented auto size
                manual_sampled = _sample_with_ratio(manual_df, pos_ratio=1, neg_ratio=2, random_state=42)
                if max_manual_size and len(manual_sampled) > max_manual_size:
                    # Further sample manual to match the cap while keeping 1:2 ratio
                    target_pos = max_manual_size // 3
                    target_neg = max_manual_size - target_pos
                    manual_pos_df = manual_sampled[manual_sampled["label"] == "TRUE"]
                    manual_neg_df = manual_sampled[manual_sampled["label"] == "FALSE"]
                    sampled_pos_df = manual_pos_df.sample(n=min(target_pos, len(manual_pos_df)), random_state=42)
                    sampled_neg_df = manual_neg_df.sample(n=min(target_neg, len(manual_neg_df)), random_state=42)
                    manual_sampled = pd.concat([sampled_pos_df, sampled_neg_df], ignore_index=True)
                    manual_sampled = manual_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

                sampled_pos = (manual_sampled["label"] == "TRUE").sum()
                sampled_neg = (manual_sampled["label"] == "FALSE").sum()
                print(f"    Manual (1:2 sampled, capped): {len(manual_sampled)} pairs ({sampled_pos} pos, {sampled_neg} neg)")

                model_types = ["logreg", "rf", "gb"]
                if HAS_XGBOOST:
                    model_types.append("xgb")

                for model_type in model_types:
                    # Train on MANUAL (curated) data - using stratified sample
                    try:
                        clf_manual, extractor_manual = train_matcher_from_df(
                            manual_sampled, df_left, df_right, model_type, id_column
                        )
                        metrics_manual = _evaluate_single_matcher(
                            test_file, clf_manual, extractor_manual, df_left, df_right, id_column
                        )
                        training_comparison_results.append({
                            "datasets": datasets_str,
                            "train_set": "manual",
                            "model": model_type,
                            "train_total": len(manual_sampled),
                            "train_pos": sampled_pos,
                            "train_neg": sampled_neg,
                            "f1": metrics_manual["f1"],
                            "precision": metrics_manual["precision"],
                            "recall": metrics_manual["recall"],
                        })
                        print(f"    {model_type} (manual): F1={metrics_manual['f1']:.3f}")
                    except Exception as e:
                        print(f"    {model_type} (manual): failed - {e}")

                    # Train on each AUTO (pipeline) dataset (using 1:2 sampled versions)
                    for auto_name, auto_df in auto_datasets_sampled:
                        try:
                            auto_pos = (auto_df["label"] == "TRUE").sum()
                            auto_neg = (auto_df["label"] == "FALSE").sum()
                            clf_auto, extractor_auto = train_matcher_from_df(
                                auto_df, df_left, df_right, model_type, id_column
                            )
                            metrics_auto = _evaluate_single_matcher(
                                test_file, clf_auto, extractor_auto, df_left, df_right, id_column
                            )
                            training_comparison_results.append({
                                "datasets": datasets_str,
                                "train_set": auto_name,
                                "model": model_type,
                                "train_total": len(auto_df),
                                "train_pos": auto_pos,
                                "train_neg": auto_neg,
                                "f1": metrics_auto["f1"],
                                "precision": metrics_auto["precision"],
                                "recall": metrics_auto["recall"],
                            })
                            print(f"    {model_type} ({auto_name}): F1={metrics_auto['f1']:.3f}")
                        except Exception as e:
                            print(f"    {model_type} ({auto_name}): failed - {e}")

            except Exception as e:
                print(f"  Error processing {manual_train_file.name}: {e}")
                import traceback
                traceback.print_exc()

        # Save training data comparison results
        if training_comparison_results:
            comparison_df = pd.DataFrame(training_comparison_results)
            comparison_path = test_output_dir / "training_data_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\nSaved training data comparison to: {comparison_path}")

    return test_evaluation_results, fusion_eval_results
