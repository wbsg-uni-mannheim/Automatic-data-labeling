from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionConfig:
    id_column: str = "id"
    trust_key: str = "trust_score"
    include_singletons: bool = True
    debug: bool = True
    use_llm: bool = True
    force_replan: bool = False


def _is_missing_like(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        if s.lower() == "nan":
            return True
    return False


def _sample_values(datasets: Sequence[pd.DataFrame], col: str, *, sample_n: int = 200) -> List[Any]:
    out: list[Any] = []
    for df in datasets:
        if col not in df.columns:
            continue
        try:
            s = df[col]
        except Exception:
            continue
        try:
            vals = s.dropna().head(sample_n).tolist()
        except Exception:
            vals = []
        for v in vals:
            if _is_missing_like(v):
                continue
            out.append(v)
            if len(out) >= sample_n:
                return out
    return out


def _infer_separator(values: List[Any]) -> Optional[str]:
    if not values:
        return None
    candidates = [",", ";", "|"]
    counts = {c: 0 for c in candidates}
    total = 0
    for v in values:
        if not isinstance(v, str):
            continue
        total += 1
        for c in candidates:
            if c in v:
                counts[c] += 1
    if total == 0:
        return None
    best = max(counts.items(), key=lambda kv: kv[1])
    if best[1] <= 0:
        return None
    # For very small samples, accept a single occurrence.
    if total < 5:
        return best[0]
    return best[0] if best[1] >= max(2, int(0.2 * total)) else None


def _looks_boolean(values: List[Any]) -> bool:
    if not values:
        return False
    tokens = {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}
    ok = 0
    total = 0
    for v in values:
        if isinstance(v, bool):
            ok += 1
            total += 1
            continue
        if isinstance(v, (int, float)) and not pd.isna(v):
            total += 1
            if v in (0, 1):
                ok += 1
            continue
        if isinstance(v, str):
            total += 1
            if v.strip().lower() in tokens:
                ok += 1
    return total > 0 and (ok / total) >= 0.9


def _looks_numeric(values: List[Any]) -> bool:
    if not values:
        return False
    total = 0
    ok = 0
    for v in values:
        if isinstance(v, (int, float)) and not pd.isna(v):
            total += 1
            ok += 1
            continue
        if isinstance(v, str):
            total += 1
            x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            if not pd.isna(x):
                ok += 1
    return total > 0 and (ok / total) >= 0.9


def _looks_date(values: List[Any]) -> bool:
    if not values:
        return False
    total = 0
    ok = 0
    for v in values:
        if isinstance(v, (pd.Timestamp,)):
            total += 1
            ok += 1
            continue
        if isinstance(v, str):
            total += 1
            x = pd.to_datetime(pd.Series([v]), errors="coerce").iloc[0]
            if not pd.isna(x):
                ok += 1
    return total > 0 and (ok / total) >= 0.85


def _looks_list(values: List[Any]) -> Tuple[bool, Optional[str]]:
    if not values:
        return False, None
    has_list = any(isinstance(v, (list, tuple, set)) for v in values)
    sep = _infer_separator(values)
    if has_list:
        # Mixed list + string-lists: still pass a separator so union can split string values.
        return True, sep
    return (sep is not None), sep


def _identifier_like(col: str, *, id_column: str) -> bool:
    c = col.lower()
    if c == id_column.lower() or c == "_id":
        return False
    if c.endswith("_id"):
        return True
    hints = ("isbn", "issn", "ean", "gtin", "upc", "sku", "vat", "ein", "doi", "imdb", "tmdb", "asin")
    return any(h in c for h in hints)


def _compute_trust_scores(datasets: Dict[str, pd.DataFrame], *, id_column: str) -> Dict[str, float]:
    """
    Heuristic trust: higher average non-null coverage => higher trust.
    Scaled to roughly [1.0, 3.0] for convenience.
    """
    scores: Dict[str, float] = {}
    coverages: Dict[str, float] = {}
    for name, df in datasets.items():
        cols = [c for c in df.columns if c not in {id_column, "_id"} and not c.startswith("_fusion_")]
        if not cols:
            coverages[name] = 0.0
            continue
        cov = float(df[cols].notna().mean().mean()) if len(df) else 0.0
        coverages[name] = cov
    if not coverages:
        return scores
    vals = list(coverages.values())
    lo = min(vals)
    hi = max(vals)
    for name, cov in coverages.items():
        if hi > lo:
            scaled = (cov - lo) / (hi - lo)
        else:
            scaled = 0.0
        scores[name] = 1.0 + 2.0 * float(scaled)
    return scores


def _resolver_registry() -> Dict[str, Any]:
    from PyDI import fusion as fusion_mod

    return {
        # Strings
        "longest_string": fusion_mod.longest_string,
        "shortest_string": fusion_mod.shortest_string,
        "most_complete": fusion_mod.most_complete,
        # Numerics
        "average": fusion_mod.average,
        "median": fusion_mod.median,
        "maximum": fusion_mod.maximum,
        "minimum": fusion_mod.minimum,
        "sum_values": fusion_mod.sum_values,
        # Dates
        "most_recent": fusion_mod.most_recent,
        "earliest": fusion_mod.earliest,
        # Lists
        "union": fusion_mod.union,
        "intersection": fusion_mod.intersection,
        "intersection_k_sources": fusion_mod.intersection_k_sources,
        # Source-aware / general
        "voting": fusion_mod.voting,
        "weighted_voting": fusion_mod.weighted_voting,
        "favour_sources": fusion_mod.favour_sources,
        "prefer_higher_trust": fusion_mod.prefer_higher_trust,
        "random_value": fusion_mod.random_value,
    }


def _heuristic_recommendation(
    datasets: Sequence[pd.DataFrame], col: str, *, config: FusionConfig
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Returns (resolver_name, resolver_kwargs, diagnostics_for_llm).
    """
    values = _sample_values(datasets, col, sample_n=200)
    is_list, sep = _looks_list(values)
    diag: Dict[str, Any] = {
        "attribute": col,
        "inferred_list": bool(is_list),
        "inferred_separator": sep,
        "inferred_boolean": _looks_boolean(values),
        "inferred_numeric": _looks_numeric(values),
        "inferred_date": _looks_date(values),
        "identifier_like": _identifier_like(col, id_column=config.id_column),
        "sample_values": [v if isinstance(v, (int, float, bool)) else str(v)[:120] for v in values[:8]],
    }

    if is_list:
        kwargs = {"separator": sep} if sep else {}
        return "union", kwargs, diag
    if diag["inferred_boolean"]:
        return "voting", {}, diag
    if diag["inferred_numeric"]:
        return "median", {}, diag
    if diag["identifier_like"]:
        return "prefer_higher_trust", {"trust_key": config.trust_key}, diag
    if "date" in col.lower() or diag["inferred_date"]:
        return "prefer_higher_trust", {"trust_key": config.trust_key}, diag
    if any(k in col.lower() for k in ("title", "name")):
        return "longest_string", {}, diag
    return "most_complete", {}, diag


def select_fusion_rules_with_llm(
    datasets: Dict[str, pd.DataFrame],
    *,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ask an LLM to choose exactly one conflict resolution function per attribute.

    Returns mapping:
      {attribute: {"resolver": "<name>", "kwargs": {...}}}

    Caches the result to disk when output_dir is provided.
    """
    from langchain_core.messages import HumanMessage

    out_dir = Path(output_dir) if output_dir is not None else None
    cache_path = out_dir / "fusion_rules.json" if out_dir is not None else None
    prompt_path = out_dir / "fusion_rules_prompt.txt" if out_dir is not None else None

    if cache_path is not None and cache_path.exists() and not config.force_replan:
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached, dict) and "attribute_rules" in cached and isinstance(cached["attribute_rules"], dict):
                try:
                    main_entity = cached.get("main_entity")
                    rules = cached.get("attribute_rules") or {}
                    by_resolver: Dict[str, int] = {}
                    unions: List[Dict[str, Any]] = []
                    for attr, spec in rules.items():
                        if not isinstance(spec, dict):
                            continue
                        rname = str(spec.get("resolver") or "")
                        by_resolver[rname] = by_resolver.get(rname, 0) + 1
                        if rname == "union":
                            kwargs = spec.get("kwargs") or {}
                            unions.append({"attribute": attr, "separator": kwargs.get("separator")})
                    logger.info(
                        "Loaded cached LLM fusion rules (main_entity=%s, resolvers=%s, union_attrs=%s) from %s",
                        main_entity,
                        dict(sorted(by_resolver.items(), key=lambda kv: (-kv[1], kv[0]))),
                        unions,
                        cache_path,
                    )
                except Exception:
                    pass
                return cached["attribute_rules"]
        except Exception:
            pass

    ds_list = list(datasets.values())
    all_cols: set[str] = set()
    for df in ds_list:
        all_cols.update(df.columns)

    excluded = {config.id_column, "_id"}
    cols = [c for c in sorted(all_cols) if c not in excluded and not str(c).startswith("_fusion_")]

    # Build per-attribute diagnostics + heuristic starting plan.
    per_attr: List[Dict[str, Any]] = []
    heuristic: Dict[str, Dict[str, Any]] = {}
    for col in cols:
        resolver, kwargs, diag = _heuristic_recommendation(ds_list, col, config=config)
        heuristic[col] = {"resolver": resolver, "kwargs": kwargs}
        per_attr.append(diag)

    resolver_names = sorted(_resolver_registry().keys())
    dataset_overview = []
    for name, df in datasets.items():
        dataset_overview.append(
            {
                "dataset_name": name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "trust_score": df.attrs.get(config.trust_key),
            }
        )

    prompt = f"""We are doing DATA FUSION: merge multiple records that refer to the same real-world entity.

Task:
1) Infer what the MAIN ENTITY is from dataset names and attribute names.
2) For EACH attribute, choose EXACTLY ONE conflict resolution function (resolver) from the allowed list.
3) The main goal is to correctly identify list-like attributes and choose a list resolver (usually 'union').

Allowed resolvers (choose only from this list):
{resolver_names}

Resolver guidance:
- Lists/Sets: union, intersection, intersection_k_sources (if you choose union for string-lists, you may set kwargs.separator to ',', ';', or '|')
- Strings: most_complete, longest_string, shortest_string
- Numerics: median, average, maximum, minimum, sum_values
- Dates: most_recent, earliest
- Source-aware: prefer_higher_trust (use kwargs.trust_key='{config.trust_key}'), voting, weighted_voting, favour_sources, random_value

Starting point (heuristic plan you can override):
{json.dumps(heuristic, indent=2, sort_keys=True)}

Attribute diagnostics (to help spot list-like attributes and separators):
{pd.DataFrame(per_attr).to_string(index=False)}

Dataset overview (trust_score may be missing; you can still use prefer_higher_trust if it makes sense):
{json.dumps(dataset_overview, indent=2)}

Return ONLY strict JSON in this schema:
{{
  "main_entity": "<string>",
  "attribute_rules": {{
    "<attribute>": {{"resolver": "<resolver_name>", "kwargs": {{...}}}},
    ...
  }}
}}
Notes:
- Include every attribute listed in the heuristic plan.
- Use kwargs only when needed. Do not invent new resolvers.
"""

    if prompt_path is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt)
        except Exception:
            pass

    response = chat_model.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", str(response)).strip()

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM fusion rule selection did not return valid JSON: {e}")

    main_entity = parsed.get("main_entity")
    rules = parsed.get("attribute_rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("LLM fusion rule selection returned no attribute_rules")

    # Validate + normalize.
    registry = _resolver_registry()
    out_rules: Dict[str, Dict[str, Any]] = {}
    for attr, spec in rules.items():
        if not isinstance(attr, str) or attr not in heuristic:
            continue
        if not isinstance(spec, dict):
            continue
        resolver_name = str(spec.get("resolver") or "").strip()
        kwargs = spec.get("kwargs") or {}
        if resolver_name not in registry:
            resolver_name = heuristic[attr]["resolver"]
            kwargs = heuristic[attr].get("kwargs") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        # Ensure trust_key is present when requested.
        if resolver_name == "prefer_higher_trust":
            kwargs = dict(kwargs)
            kwargs.setdefault("trust_key", config.trust_key)
        out_rules[attr] = {"resolver": resolver_name, "kwargs": kwargs}

    # Ensure we return something for every attribute (fallback to heuristic).
    for attr in heuristic.keys():
        if attr not in out_rules:
            out_rules[attr] = heuristic[attr]

    if cache_path is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "main_entity": main_entity,
                        "attribute_rules": out_rules,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
        except Exception:
            pass

    # Log a compact summary (full details are in fusion_rules.json when output_dir is provided).
    try:
        by_resolver: Dict[str, int] = {}
        unions: List[Dict[str, Any]] = []
        for attr, spec in out_rules.items():
            rname = str(spec.get("resolver") or "")
            by_resolver[rname] = by_resolver.get(rname, 0) + 1
            if rname == "union":
                kwargs = spec.get("kwargs") or {}
                unions.append({"attribute": attr, "separator": kwargs.get("separator")})
        logger.info(
            "LLM selected fusion rules (main_entity=%s, resolvers=%s, union_attrs=%s).",
            main_entity,
            dict(sorted(by_resolver.items(), key=lambda kv: (-kv[1], kv[0]))),
            unions,
        )
        if cache_path is not None:
            logger.info("LLM fusion rules saved to %s", cache_path)
        if prompt_path is not None and prompt_path.exists():
            logger.info("LLM fusion prompt saved to %s", prompt_path)
    except Exception:
        pass

    return out_rules


def auto_build_fusion_strategy(
    datasets: Dict[str, pd.DataFrame],
    *,
    config: FusionConfig = FusionConfig(),
    strategy_name: str = "auto_fusion_strategy",
    exclude_columns: Optional[Iterable[str]] = None,
) -> "DataFusionStrategy":
    """
    Build a DataFusionStrategy automatically based on inferred attribute types.

    - list-like: union (auto separator for string lists)
    - boolean-like: voting
    - numeric-like: median
    - date-like: prefer_higher_trust when trust varies, else voting
    - identifier-like: prefer_higher_trust when trust varies, else voting
    - string default: most_complete
    """
    from PyDI.fusion import (
        DataFusionStrategy,
        longest_string,
        median,
        most_complete,
        prefer_higher_trust,
        union,
        voting,
    )

    ds_list = list(datasets.values())
    id_column = config.id_column
    excluded = set(exclude_columns or [])
    excluded.update({id_column, "_id"})

    # Determine whether trust is informative (not all equal).
    trust_vals = []
    for df in ds_list:
        v = df.attrs.get(config.trust_key)
        if v is not None:
            try:
                trust_vals.append(float(v))
            except Exception:
                pass
    trust_informative = len(set(trust_vals)) > 1

    all_cols: set[str] = set()
    for df in ds_list:
        all_cols.update(df.columns)

    strategy = DataFusionStrategy(strategy_name)

    for col in sorted(all_cols):
        if col in excluded or col.startswith("_fusion_"):
            continue

        values = _sample_values(ds_list, col, sample_n=200)

        is_list, sep = _looks_list(values)
        if is_list:
            kwargs = {"separator": sep} if sep else {}
            strategy.add_attribute_fuser(col, union, **kwargs)
            continue

        if _looks_boolean(values):
            strategy.add_attribute_fuser(col, voting)
            continue

        if _looks_numeric(values):
            strategy.add_attribute_fuser(col, median)
            continue

        # Prefer trust for IDs and dates if possible; otherwise vote.
        if _identifier_like(col, id_column=id_column):
            if trust_informative:
                strategy.add_attribute_fuser(col, prefer_higher_trust, trust_key=config.trust_key)
            else:
                strategy.add_attribute_fuser(col, voting)
            continue

        if "date" in col.lower() or _looks_date(values):
            if trust_informative:
                strategy.add_attribute_fuser(col, prefer_higher_trust, trust_key=config.trust_key)
            else:
                strategy.add_attribute_fuser(col, voting)
            continue

        # Text default: most_complete; for titles/names, longest_string tends to work well.
        if any(k in col.lower() for k in ("title", "name")):
            strategy.add_attribute_fuser(col, longest_string)
        else:
            strategy.add_attribute_fuser(col, most_complete)

    return strategy


def run_data_fusion(
    datasets: Dict[str, pd.DataFrame],
    *,
    correspondences: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    config: FusionConfig = FusionConfig(),
    strategy: Optional["DataFusionStrategy"] = None,
    chat_model: Any = None,
    output_dir: Optional[Union[str, Path]] = None,
    schema_correspondences: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, "DataFusionStrategy", pd.DataFrame]:
    """
    Execute data fusion for a set of datasets + correspondences.

    Returns (fused_df, strategy_used, cluster_size_distribution).
    """
    from PyDI.fusion import DataFusionEngine

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure required dataset metadata.
    trust_scores = _compute_trust_scores(datasets, id_column=config.id_column)
    ds_list: list[pd.DataFrame] = []
    datasets_with_attrs: Dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        df2 = df.copy()
        try:
            df2.attrs = dict(df.attrs)
        except Exception:
            pass
        df2.attrs["dataset_name"] = name
        if config.trust_key not in df2.attrs:
            df2.attrs[config.trust_key] = float(trust_scores.get(name, 1.0))
        ds_list.append(df2)
        datasets_with_attrs[name] = df2

    if strategy is None:
        if config.use_llm and chat_model is not None:
            try:
                logger.info("Selecting fusion rules with LLM (cache=%s, force_replan=%s)", bool(out_dir), config.force_replan)
                rules = select_fusion_rules_with_llm(
                    datasets_with_attrs,
                    chat_model=chat_model,
                    config=config,
                    output_dir=out_dir,
                )
                from PyDI.fusion import DataFusionStrategy

                s = DataFusionStrategy("llm_fusion_strategy")
                registry = _resolver_registry()
                for attr, spec in rules.items():
                    resolver = registry.get(spec.get("resolver"))
                    if resolver is None:
                        continue
                    kwargs = spec.get("kwargs") or {}
                    s.add_attribute_fuser(attr, resolver, **kwargs)
                strategy = s
            except Exception as e:
                logger.warning(f"LLM fusion strategy selection failed; falling back to heuristics: {e}")
                strategy = auto_build_fusion_strategy(datasets_with_attrs, config=config)
        else:
            strategy = auto_build_fusion_strategy(datasets_with_attrs, config=config)

    debug_file = None
    if out_dir is not None and config.debug:
        debug_file = out_dir / "fusion_debug.jsonl"

    engine = DataFusionEngine(strategy, debug=bool(config.debug), debug_file=debug_file, debug_format="json")
    fused = engine.run(
        datasets=ds_list,
        correspondences=correspondences,
        schema_correspondences=schema_correspondences,
        id_column=config.id_column,
        include_singletons=bool(config.include_singletons),
    )

    if out_dir is not None:
        fused_path = out_dir / "fused.csv"
        fused.to_csv(fused_path, index=False)
        logger.info(f"Fused dataset saved to {fused_path}")
        try:
            # Persist a small strategy summary for reproducibility.
            rules: Dict[str, str] = {}
            try:
                for attr in sorted(strategy.get_registered_attributes()):
                    fuser = strategy.get_attribute_fuser(attr)
                    if fuser is not None:
                        rules[attr] = getattr(fuser.resolver, "__name__", repr(fuser.resolver))
            except Exception:
                rules = {}

            (out_dir / "fusion_strategy.json").write_text(
                json.dumps(
                    {
                        "strategy_name": strategy.name,
                        "id_column": config.id_column,
                        "trust_key": config.trust_key,
                        "include_singletons": config.include_singletons,
                        "trust_scores": trust_scores,
                        "rules": rules,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
        except Exception:
            pass

    return fused, strategy, getattr(engine, "cluster_size_distribution", pd.DataFrame())
