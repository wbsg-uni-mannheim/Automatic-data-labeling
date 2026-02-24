"""
Matching optimization using validation sets.

This module provides functions to optimize entity matching parameters
by evaluating different matchers against labeled validation data.

Supports multiple matcher families:
- Rule-based matching with configurable similarity functions
- ML-based matching with various classifiers
- LLM-based matching (optional, expensive)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..entitymatching.evaluation import EntityMatchingEvaluator
from ..utils.similarity_registry import SimilarityRegistry
from .labeled_set_generation import drop_overlapping_pairs

logger = logging.getLogger(__name__)


def _labels_to_binary(labels: pd.Series) -> pd.Series:
    """Normalize a label column to 0/1 integers (1=match).

    Supports various label formats: TRUE/FALSE, T/F, 1/0, YES/NO, Y/N,
    and boolean/numeric types.
    """
    if labels is None:
        raise ValueError("labels cannot be None")

    if pd.api.types.is_bool_dtype(labels):
        return labels.astype(int)

    # Numeric (already 0/1)
    if pd.api.types.is_numeric_dtype(labels):
        return (labels.astype(float) > 0).astype(int)

    # String-like
    s = labels.astype("string").str.strip().str.upper()
    true_values = {"TRUE", "T", "1", "YES", "Y"}
    false_values = {"FALSE", "F", "0", "NO", "N"}
    out = pd.Series(0, index=s.index, dtype="int64")
    out[s.isin(true_values)] = 1
    out[s.isin(false_values)] = 0

    unknown = ~(s.isin(true_values) | s.isin(false_values))
    if bool(unknown.any()):
        bad = sorted(set(s[unknown].dropna().tolist()))
        raise ValueError(f"Unrecognized label values: {bad[:10]}")

    return out


def _auto_select_matching_columns(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    id_column: str,
    max_columns: int = 6,
) -> List[str]:
    """Pick a small set of high-coverage common columns for matching features."""
    common = [c for c in df_left.columns if c in df_right.columns and c != id_column]
    # Exclude ID-like columns (heuristic)
    common = [c for c in common if "id" not in c.lower()]
    if not common:
        return []

    scored: list[tuple[float, str]] = []
    for col in common:
        left_non_null = float(df_left[col].notna().mean()) if len(df_left) else 0.0
        right_non_null = float(df_right[col].notna().mean()) if len(df_right) else 0.0
        score = min(left_non_null, right_non_null)
        scored.append((score, col))

    scored.sort(reverse=True)
    return [col for _, col in scored[: max_columns]]


def optimize_matching(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    validation_set: pd.DataFrame,
    *,
    id_column: str = "id",
    matching_columns: Optional[List[str]] = None,
    thresholds: Optional[Sequence[float]] = None,
    include_rule_based: bool = True,
    rule_string_similarity_functions: Optional[Sequence[str]] = None,
    rule_tokenizations: Optional[Sequence[str]] = None,
    include_ml_based: bool = True,
    training_set: Optional[pd.DataFrame] = None,
    training_set_info: Optional[Dict[str, Any]] = None,
    ml_classifiers: Optional[Sequence[Tuple[str, Any]]] = None,
    ml_use_probabilities: bool = True,
    include_llm_based: bool = False,
    llm_chat_model: Any = None,
    llm_fields: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Try multiple matcher families and pick the best F1 on the validation set.

    Notes
    -----
    - Evaluation is done on the *validation pairs* (id1, id2) only.
    - For ML-based matching, you must provide a labeled training_set (or generate
      one upstream) with columns [id1, id2, label]. Exact (id1, id2) overlaps
      with validation_set are removed automatically.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    validation_set : pd.DataFrame
        Labeled validation set with columns [id1, id2, label]
    id_column : str
        ID column name
    matching_columns : list[str], optional
        Columns to use for matching features. Auto-selected if None.
    thresholds : sequence of float, optional
        Thresholds to test. Default: 0.00 to 1.00 in steps of 0.05.
    include_rule_based : bool
        Whether to test rule-based matchers
    rule_string_similarity_functions : sequence of str, optional
        Similarity functions to test for rule-based matching
    rule_tokenizations : sequence of str, optional
        Tokenization strategies to try for rule-based string comparators.
        Only applies to tokenizable similarity functions (e.g., jaccard, cosine).
        Default: ["word", "char"].
    include_ml_based : bool
        Whether to test ML-based matchers
    training_set : pd.DataFrame, optional
        Labeled training set for ML matchers
    training_set_info : dict, optional
        Info to generate training set if not provided
    ml_classifiers : sequence of (name, classifier), optional
        Classifiers to test. Default: LogisticRegression, RandomForest.
    ml_use_probabilities : bool
        Whether to use probability scores from classifiers
    include_llm_based : bool
        Whether to test LLM-based matcher (expensive)
    llm_chat_model : BaseChatModel, optional
        Chat model for LLM matching
    llm_fields : list[str], optional
        Fields to show LLM for matching
    out_dir : Path, optional
        Directory to save results
    out_path : Path, optional
        Exact CSV path to save results. If provided, takes precedence over out_dir.

    Returns
    -------
    dict
        Best configuration and all results
    """
    from ..entitymatching import (
        DateComparator,
        FeatureExtractor,
        LLMBasedMatcher,
        MLBasedMatcher,
        NumericComparator,
        RuleBasedMatcher,
        StringComparator,
    )
    from .labeled_set_generation import load_or_generate_training_set

    if validation_set.empty:
        raise ValueError("validation_set is empty")
    for col in ["id1", "id2"]:
        if col not in validation_set.columns:
            raise ValueError(f"validation_set missing required column: {col}")

    eval_pairs = validation_set[["id1", "id2"]].drop_duplicates().reset_index(drop=True)

    if thresholds is None:
        thresholds = [i / 20 for i in range(0, 21)]  # 0.00..1.00
    thresholds = [float(t) for t in thresholds]

    if matching_columns is None:
        matching_columns = _auto_select_matching_columns(
            df_left, df_right, id_column=id_column
        )
        logger.info(f"Auto-selected matching columns: {matching_columns}")

    default_rule_tokenizations = ["word", "char"]
    if rule_tokenizations is None:
        rule_tokenizations = default_rule_tokenizations
    rule_tokenizations = [str(t) for t in rule_tokenizations]

    def _iter_string_similarity_settings(
        sim_fns: Sequence[str],
        *,
        tokenizations: Sequence[str],
    ) -> List[Tuple[str, Optional[str]]]:
        """Return (sim_fn, tokenization) settings to try.

        For tokenizable functions, we try multiple tokenizations.
        For non-tokenizable functions, tokenization is None (use default char-level).

        Note: monge_elkan is excluded from char tokenization because it's O(n*m)
        where n and m are token counts, making char-level prohibitively slow
        (~200 calls/sec vs ~5000 calls/sec for word tokenization).
        """
        # Functions that should NOT use char tokenization (too slow)
        skip_char_tokenization = {"monge_elkan"}

        settings: List[Tuple[str, Optional[str]]] = []
        for fn in sim_fns:
            fn = str(fn)
            if SimilarityRegistry.is_tokenizable(fn):
                for tok in tokenizations:
                    # Skip char tokenization for slow O(n*m) algorithms
                    if tok == "char" and fn in skip_char_tokenization:
                        continue
                    settings.append((fn, str(tok)))
            else:
                settings.append((fn, None))
        # Stable de-dupe while preserving order.
        seen: set[Tuple[str, Optional[str]]] = set()
        uniq: List[Tuple[str, Optional[str]]] = []
        for s in settings:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return uniq

    def _build_comparators(
        *,
        string_similarity_function: str,
        string_tokenization: Optional[str] = None,
        date_max_days_difference: int = 365 * 2,
    ) -> List[Any]:
        cols = matching_columns or []
        comparators: List[Any] = []
        for col in cols:
            if col == id_column:
                continue
            if col not in df_left.columns or col not in df_right.columns:
                continue

            left_dtype = df_left[col].dtype
            right_dtype = df_right[col].dtype
            col_lower = col.lower()

            # Datetime-like
            if (
                "date" in col_lower
                or "birthday" in col_lower
                or pd.api.types.is_datetime64_any_dtype(left_dtype)
                or pd.api.types.is_datetime64_any_dtype(right_dtype)
            ):
                comparators.append(DateComparator(col, max_days_difference=date_max_days_difference))
                continue

            # Numeric/bool
            if pd.api.types.is_bool_dtype(left_dtype) and pd.api.types.is_bool_dtype(right_dtype):
                comparators.append(lambda r1, r2, _c=col: float(bool(r1.get(_c)) == bool(r2.get(_c))))
                continue
            if pd.api.types.is_numeric_dtype(left_dtype) and pd.api.types.is_numeric_dtype(right_dtype):
                comparators.append(NumericComparator(col, method="relative_difference", max_difference=0.25))
                continue

            # Default: string comparator (treat as text)
            comparators.append(
                StringComparator(
                    col,
                    string_similarity_function,
                    tokenization=string_tokenization,
                    preprocess=str.lower,
                )
            )

        return comparators

    results: List[Dict[str, Any]] = []
    all_artifacts: List[Dict[str, Any]] = []  # Store artifacts for all results
    best: Optional[Dict[str, Any]] = None
    best_artifacts: Dict[str, Any] = {}

    def _record(
        *,
        matcher_name: str,
        config: Dict[str, Any],
        threshold: float,
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal best, best_artifacts
        row = {
            "matcher": matcher_name,
            "threshold": float(threshold),
            "f1": float(metrics.get("f1", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "true_positives": int(metrics.get("true_positives", 0)),
            "false_positives": int(metrics.get("false_positives", 0)),
            "false_negatives": int(metrics.get("false_negatives", 0)),
            **config,
        }
        results.append(row)
        all_artifacts.append(artifacts or {})  # Track artifacts for each result
        if best is None or row["f1"] > best["f1"]:
            best = row
            best_artifacts = artifacts or {}

    # Rule-based matcher
    if include_rule_based:
        if rule_string_similarity_functions is None:
            # Full recommended list by default (configurable via rule_string_similarity_functions).
            rule_string_similarity_functions = SimilarityRegistry.get_recommended_functions("entity_matching")

        for sim_fn, tok in _iter_string_similarity_settings(
            rule_string_similarity_functions,
            tokenizations=rule_tokenizations,
        ):
            try:
                comparators = _build_comparators(
                    string_similarity_function=sim_fn,
                    string_tokenization=tok,
                )
                if not comparators:
                    logger.warning("No comparators selected; skipping RuleBasedMatcher")
                    break

                matcher = RuleBasedMatcher()
                corr = matcher.match(
                    df_left=df_left,
                    df_right=df_right,
                    candidates=eval_pairs,
                    id_column=id_column,
                    comparators=comparators,
                    threshold=0.0,  # keep all scored pairs for threshold sweep
                )

                for thr in thresholds:
                    metrics = EntityMatchingEvaluator.evaluate_matching(
                        correspondences=corr,
                        test_pairs=validation_set,
                        threshold=float(thr),
                        matcher_instance=matcher,
                    )
                    _record(
                        matcher_name="RuleBasedMatcher",
                        config={
                            "string_similarity_function": sim_fn,
                            "string_tokenization": tok or "",
                            "n_comparators": len(comparators),
                        },
                        threshold=float(thr),
                        metrics=metrics,
                        artifacts={"matcher": matcher, "comparators": comparators, "corr": corr},
                    )
            except Exception as e:
                logger.warning(f"RuleBasedMatcher failed for sim_fn={sim_fn}, tok={tok}: {e}")

    # ML-based matcher
    if include_ml_based:
        if training_set is None:
            if training_set_info is None:
                raise ValueError(
                    "include_ml_based=True requires training_set (labeled pairs) to be provided, "
                    "or pass training_set_info to generate it via load_or_generate_training_set."
                )

            required_keys = {"left_name", "right_name", "chat_model", "output_dir"}
            missing = required_keys - set(training_set_info.keys())
            if missing:
                raise ValueError(f"training_set_info missing required keys: {sorted(missing)}")

            training_set = load_or_generate_training_set(
                df_left=df_left,
                df_right=df_right,
                left_name=str(training_set_info["left_name"]),
                right_name=str(training_set_info["right_name"]),
                chat_model=training_set_info["chat_model"],
                output_dir=Path(training_set_info["output_dir"]),
                id_column=id_column,
                target_size=int(training_set_info.get("target_size", 500)),
                target_positives=int(training_set_info.get("target_positives", 150)),
                similarity_threshold=float(training_set_info.get("similarity_threshold", 0.3)),
                force_regenerate=bool(training_set_info.get("force_regenerate", False)),
                exclude_pairs=validation_set,
                blocker_spec=training_set_info.get("blocker_spec"),
                max_candidate_pairs=training_set_info.get("max_candidate_pairs"),
            )

        train_df = training_set.copy()
        train_df = drop_overlapping_pairs(train_df, exclude_pairs=validation_set)
        train_df = train_df.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)
        if train_df.empty:
            raise ValueError("training_set became empty after removing overlaps with validation_set")

        if "label" not in train_df.columns:
            raise ValueError("training_set missing required column: label")

        y_train = _labels_to_binary(train_df["label"])
        pairs_train = train_df[["id1", "id2"]]

        # Build a feature extractor with a small, robust set of string similarity features.
        feature_similarity_functions = SimilarityRegistry.get_recommended_functions("entity_matching")
        feature_functions: List[Any] = []
        for sim_fn, tok in _iter_string_similarity_settings(
            feature_similarity_functions,
            tokenizations=rule_tokenizations,
        ):
            feature_functions.extend(
                _build_comparators(
                    string_similarity_function=sim_fn,
                    string_tokenization=tok,
                )
            )
        # Deduplicate callables by repr/name to avoid exact duplicates.
        uniq_feature_functions: List[Any] = []
        seen_names: set[str] = set()
        for f in feature_functions:
            key = getattr(f, "name", None) or getattr(f, "__name__", None) or repr(f)
            if key in seen_names:
                continue
            seen_names.add(key)
            uniq_feature_functions.append(f)

        extractor = FeatureExtractor(uniq_feature_functions)

        train_features = extractor.create_features(
            df_left=df_left,
            df_right=df_right,
            pairs=pairs_train,
            id_column=id_column,
            labels=y_train,
        )

        X_train = train_features.drop(columns=["label", "id1", "id2"], errors="ignore")
        y_train2 = train_features["label"].astype(int)

        if len(set(y_train2.tolist())) < 2:
            raise ValueError("training_set must contain at least one positive and one negative example")

        if ml_classifiers is None:
            try:
                from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

                # Calculate scale_pos_weight for XGBoost to handle class imbalance
                # scale_pos_weight = number of negative samples / number of positive samples
                n_positive = int(y_train2.sum())
                n_negative = len(y_train2) - n_positive
                scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

                # Define parameter grids for hyperparameter optimization
                param_grids: Dict[str, Dict[str, Any]] = {
                    "logreg": {
                        "C": [0.01, 0.1, 1.0, 10.0],
                        "penalty": ["l2"],
                        "solver": ["lbfgs", "saga"],
                    },
                    "rf": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                    "extra_trees": {
                        "n_estimators": [200, 400, 600],
                        "max_depth": [None, 15, 30],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2],
                    },
                    "gbdt": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0],
                    },
                    "hist_gbdt": {
                        "max_iter": [100, 200],
                        "learning_rate": [0.05, 0.1, 0.2],
                        "max_depth": [None, 10, 20],
                        "min_samples_leaf": [20, 50],
                    },
                    "xgboost": {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 6, 9],
                        "learning_rate": [0.05, 0.1, 0.2],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0],
                    },
                }

                # Base classifiers (will be wrapped with RandomizedSearchCV)
                base_classifiers: List[Tuple[str, Any, Dict[str, Any]]] = [
                    ("logreg", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), param_grids["logreg"]),
                    ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"), param_grids["rf"]),
                    ("gbdt", GradientBoostingClassifier(random_state=42), param_grids["gbdt"]),
                ]

                # HistGradientBoosting is fast on larger sets but may not exist on older sklearn.
                try:
                    from sklearn.ensemble import HistGradientBoostingClassifier

                    base_classifiers.append(("hist_gbdt", HistGradientBoostingClassifier(random_state=42), param_grids["hist_gbdt"]))
                except Exception:
                    logger.info("HistGradientBoostingClassifier not available; skipping")

                # Try to add XGBoost if available
                try:
                    from xgboost import XGBClassifier

                    base_classifiers.append((
                        "xgboost",
                        XGBClassifier(
                            scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            eval_metric="logloss",
                            use_label_encoder=False,
                        ),
                        param_grids["xgboost"],
                    ))
                except ImportError:
                    logger.info("XGBoost not available; skipping XGBClassifier")

                # Perform hyperparameter optimization using RandomizedSearchCV
                cv = StratifiedKFold(n_splits=min(5, n_positive, n_negative), shuffle=True, random_state=42)
                ml_classifiers = []

                for clf_name, base_clf, param_grid in base_classifiers:
                    try:
                        # Calculate number of parameter combinations
                        n_combinations = 1
                        for values in param_grid.values():
                            n_combinations *= len(values)
                        n_iter = min(20, n_combinations)  # Cap at 20 iterations

                        search = RandomizedSearchCV(
                            base_clf,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            cv=cv,
                            scoring="f1",
                            n_jobs=-1,
                            random_state=42,
                            error_score="raise",
                        )
                        search.fit(X_train, y_train2)
                        best_clf = search.best_estimator_
                        logger.info(
                            f"Hyperparameter optimization for {clf_name}: "
                            f"best_score={search.best_score_:.3f}, best_params={search.best_params_}"
                        )
                        ml_classifiers.append((clf_name, best_clf))
                    except Exception as e:
                        logger.warning(f"Hyperparameter optimization failed for {clf_name}: {e}; using defaults")
                        # Fall back to fitting base classifier with defaults
                        base_clf.fit(X_train, y_train2)
                        ml_classifiers.append((clf_name, base_clf))

            except Exception as e:
                raise RuntimeError(f"Failed to import scikit-learn classifiers: {e}")

        for clf_name, clf in ml_classifiers:
            try:
                # Only fit if not already fitted (user-provided classifiers need fitting)
                if not hasattr(clf, "classes_"):
                    clf.fit(X_train, y_train2)

                matcher = MLBasedMatcher(extractor)
                corr = matcher.match(
                    df_left=df_left,
                    df_right=df_right,
                    candidates=[eval_pairs],
                    id_column=id_column,
                    trained_classifier=clf,
                    threshold=0.0,  # keep all scored pairs for threshold sweep
                    use_probabilities=bool(ml_use_probabilities),
                )

                # Extract hyperparameters for logging
                clf_params = clf.get_params() if hasattr(clf, "get_params") else {}
                # Filter to only include non-default/interesting params
                interesting_params = {
                    k: v for k, v in clf_params.items()
                    if not k.startswith("_") and v is not None and k not in ("random_state", "n_jobs", "verbose")
                }

                for thr in thresholds:
                    metrics = EntityMatchingEvaluator.evaluate_matching(
                        correspondences=corr,
                        test_pairs=validation_set,
                        threshold=float(thr),
                        matcher_instance=matcher,
                    )
                    _record(
                        matcher_name="MLBasedMatcher",
                        config={
                            "classifier": clf_name,
                            "use_probabilities": bool(ml_use_probabilities),
                            "n_features": int(X_train.shape[1]),
                            "hyperparams": str(interesting_params),
                        },
                        threshold=float(thr),
                        metrics=metrics,
                        artifacts={"matcher": matcher, "classifier": clf, "feature_extractor": extractor, "corr": corr},
                    )
            except Exception as e:
                logger.warning(f"MLBasedMatcher failed for classifier={clf_name}: {e}")

        # Log best ML matcher
        ml_results = [r for r in results if r["matcher"] == "MLBasedMatcher"]
        if ml_results:
            best_ml = max(ml_results, key=lambda x: (x["f1"], x["precision"], x["recall"]))
            logger.info(
                f"Best ML matcher: {best_ml['classifier']} "
                f"(F1={best_ml['f1']:.3f}, P={best_ml['precision']:.3f}, "
                f"R={best_ml['recall']:.3f}, thr={best_ml['threshold']})"
            )

    # Optional LLM-based matcher (expensive)
    if include_llm_based:
        if llm_chat_model is None:
            raise ValueError("include_llm_based=True requires llm_chat_model")

        try:
            matcher = LLMBasedMatcher()
            pred = matcher.match(
                df_left=df_left,
                df_right=df_right,
                candidates=eval_pairs,
                id_column=id_column,
                chat_model=llm_chat_model,
                fields=llm_fields,
                generate_explanations=False,
                parse_strictness="non_match",
            )
            merged = eval_pairs.merge(pred[["id1", "id2", "match"]], on=["id1", "id2"], how="left")
            merged["match"] = merged["match"].fillna(False).astype(bool)
            corr = merged.assign(score=merged["match"].astype(float), notes="llm").loc[:, ["id1", "id2", "score", "notes"]]

            # Threshold is effectively fixed for binary outputs, but keep consistent interface.
            thr = 0.5
            metrics = EntityMatchingEvaluator.evaluate_matching(
                correspondences=corr,
                test_pairs=validation_set,
                threshold=float(thr),
                matcher_instance=matcher,
            )
            _record(
                matcher_name="LLMBasedMatcher",
                config={"fields": ",".join(llm_fields or []), "binary": True},
                threshold=float(thr),
                metrics=metrics,
                artifacts={"matcher": matcher, "corr": corr},
            )
        except Exception as e:
            logger.warning(f"LLMBasedMatcher failed: {e}")

    # Sort results by F1 score, keeping artifacts in sync
    # Create (result, artifact) pairs, sort, then unzip
    results_with_artifacts = list(zip(results, all_artifacts))
    sorted_pairs = sorted(
        results_with_artifacts,
        key=lambda x: (x[0].get("f1", 0), x[0].get("precision", 0), x[0].get("recall", 0)),
        reverse=True
    )
    sorted_results = [pair[0] for pair in sorted_pairs]
    sorted_artifacts = [pair[1] for pair in sorted_pairs]

    results_df = pd.DataFrame(sorted_results)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        logger.info(f"Matcher optimization results saved to {out_path}")
    elif out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = out_dir / "matcher_optimization.csv"
        results_df.to_csv(default_path, index=False)
        logger.info(f"Matcher optimization results saved to {default_path}")

    return {
        "best": best,
        "best_artifacts": best_artifacts,
        "all_results": results_df,
        "all_artifacts": sorted_artifacts,  # Artifacts in same order as sorted results_df
        "matching_columns": matching_columns,
        "training_set_info": training_set_info or {},
    }


__all__ = [
    "optimize_matching",
]
