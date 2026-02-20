"""
LLM-based entity matching using conversational AI models.

This module provides an LLMBasedMatcher that uses large language models to
determine if pairs of records refer to the same real-world entity.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..utils.llm import LLMCallLogger
from .base import BaseMatcher

import logging


class LLMBasedMatcher(BaseMatcher):
    """
    LLM-based entity matcher using conversational AI models.

    This matcher uses large language models to determine if pairs of records
    refer to the same real-world entity. It supports both zero-shot and few-shot
    prompting modes with configurable output formats and unified LLM logging.

    Examples
    --------
    Zero-shot matching:

    >>> from langchain_openai import ChatOpenAI
    >>> from PyDI.entitymatching import LLMBasedMatcher
    >>> import pandas as pd
    >>>
    >>> # candidates is a DataFrame with id1, id2 columns
    >>> candidates = pd.DataFrame({"id1": [1, 2], "id2": [101, 102]})
    >>>
    >>> matcher = LLMBasedMatcher()
    >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates, "_id",
    ...     chat_model=chat,
    ...     fields=["name", "address", "city"]
    ... )

    Few-shot matching with examples:

    >>> few_shots = [
    ...     (
    ...         {"name": "Acme Corp", "address": "12 Main St"},
    ...         {"name": "Acme Corporation", "address": "12 Main Street"},
    ...         '{"match": true, "explanation": "name and address match"}'
    ...     )
    ... ]
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates, "_id",
    ...     chat_model=chat,
    ...     few_shots=few_shots,
    ...     generate_explanations=True
    ... )
    """

    def __init__(self):
        """Initialize the LLM-based matcher."""
        super().__init__()
        self._llm_logger = LLMCallLogger()
        self._current_run_out_dir: Optional[Path] = None

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        id_column: str,
        chat_model: BaseChatModel,
        *,
        fields: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        few_shots: Optional[List[Tuple[Dict[str, Any],
                                       Dict[str, Any], str]]] = None,
        generate_explanations: bool = False,
        include_difficulty: bool = False,
        retries: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_parser: Optional[Callable[[str],
                                           Optional[Dict[str, Any]]]] = None,
        out_dir: str = "output/entitymatching/llm",
        debug: bool = False,
        max_concurrency: int = 1,
        rate_limit_per_sec: Optional[float] = None,
        parse_strictness: str = "skip"
    ) -> pd.DataFrame:
        """
        Match entity pairs using an LLM.

        Parameters
        ----------
        df_left : pd.DataFrame
            Left dataset with records to match.
        df_right : pd.DataFrame
            Right dataset with records to match.
        candidates : pd.DataFrame or Iterable[pd.DataFrame]
            Single DataFrame or iterable of candidate pair batches with id1, id2 columns
            representing candidate pairs to evaluate.
        id_column : str
            Name of the column containing record identifiers.
        chat_model : BaseChatModel
            LangChain chat model instance (e.g., ChatOpenAI, ChatAnthropic).
        fields : Optional[List[str]], default=None
            List of column names to include in LLM prompts. If None, auto-selects
            string-like columns up to a reasonable limit.
        system_prompt : Optional[str], default=None
            Custom system prompt. If None, uses a default entity matching prompt.
        few_shots : Optional[List[Tuple[Dict, Dict, str]]], default=None
            Few-shot examples as (left_record, right_record, expected_json) tuples.
        generate_explanations : bool, default=False
            Whether to request explanations from the LLM. When True, the LLM
            provides a brief explanation for each decision in the explanation column.
        include_difficulty : bool, default=False
            Whether to request difficulty annotations from the LLM. When True, the
            LLM will classify each decision as "simple", "medium", or "difficult".
        retries : int, default=1
            Number of retry attempts on API failures.
        temperature : Optional[float], default=None
            Deprecated: temperature is no longer passed to the chat model. Configure
            temperature on the chat model instance itself.
        max_tokens : Optional[int], default=None
            Maximum tokens in model response.
        response_parser : Optional[Callable[[str], Optional[Dict[str, Any]]]], default=None
            Optional custom parser that takes the raw response text and returns a
            dict with 'match' (bool) and optionally 'explanation' (str) and 'difficulty' (str).
            Use this when your prompt returns a different format than the default JSON.
        out_dir : str, default="output/entitymatching/llm"
            Directory for writing debug artifacts and logs.
        debug : bool, default=False
            Whether to write debug artifacts (prompts, responses, stats).
        max_concurrency : int, default=1
            Maximum concurrent API calls (not implemented yet).
        rate_limit_per_sec : Optional[float], default=None
            Rate limit for API calls per second (not implemented yet).
        parse_strictness : str, default="skip"
            How to handle JSON parsing errors: "skip" to exclude the pair, or
            "non_match" to include as non-match.

        Returns
        -------
        pd.DataFrame
            CorrespondenceSet with columns: id1, id2, match (and optionally
            explanation, difficulty based on flags).
        """
        # Validate inputs using base class
        self._validate_inputs(df_left, df_right, id_column)

        # Initialize output directory and remember for artifact writes
        self._current_run_out_dir = Path(out_dir)
        if debug:
            self._current_run_out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-select fields if not provided
        if fields is None:
            fields = self._auto_select_fields(df_left, df_right, id_column)

        # Build prompt template
        prompt_template = self._build_prompt_template(
            system_prompt, few_shots, generate_explanations, include_difficulty)

        # Normalize candidates to iterable of DataFrames
        if isinstance(candidates, pd.DataFrame):
            candidate_batches = [candidates]
        else:
            candidate_batches = list(candidates)

        # Log matching info
        self._log_matching_info(df_left, df_right, candidate_batches)

        matches = []
        pair_index = 0
        true_matches = 0
        false_matches = 0

        progress_logger = logging.getLogger("PyDI.entitymatching.llm_progress")
        try:
            total_pairs = int(sum(len(b) for b in candidate_batches if b is not None))
        except Exception:
            total_pairs = 0
        try:
            progress_every = int(os.getenv("PYDI_LLM_PROGRESS_EVERY", "10"))
        except Exception:
            progress_every = 10
        if progress_every <= 0:
            progress_every = 10

        # Process candidate batches
        batch_count = len([b for b in candidate_batches if b is not None and not b.empty])
        batch_idx = 0
        for batch in candidate_batches:
            if batch.empty:
                continue
            batch_idx += 1
            batch_true = 0
            batch_false = 0
            batch_start = time.perf_counter()

            # Validate batch has required columns
            if "id1" not in batch.columns or "id2" not in batch.columns:
                raise ValueError("Candidate DataFrame must have 'id1' and 'id2' columns")

            # Process each pair in the batch
            for _, row in batch.iterrows():
                left_id, right_id = row["id1"], row["id2"]

                # Get records
                left_record = df_left[df_left[id_column] == left_id].iloc[0]
                right_record = df_right[df_right[id_column] == right_id].iloc[0]

                # Serialize records for prompt
                left_data = self._serialize_record(left_record, fields, id_column)
                right_data = self._serialize_record(right_record, fields, id_column)

                # Try matching with retries
                match_result = self._match_pair_with_retry(
                    prompt_template, left_data, right_data, chat_model,
                    retries, temperature, max_tokens, pair_index, out_dir, debug,
                    parse_strictness, generate_explanations, include_difficulty, response_parser)

                if match_result:
                    if bool(match_result.get("match")):
                        true_matches += 1
                        batch_true += 1
                    else:
                        false_matches += 1
                        batch_false += 1
                    match_entry = {
                        "id1": left_id,
                        "id2": right_id,
                        "match": match_result["match"]
                    }
                    if generate_explanations:
                        match_entry["explanation"] = match_result.get("explanation", "")
                    if include_difficulty:
                        match_entry["difficulty"] = match_result.get("difficulty", "")
                    matches.append(match_entry)
                else:
                    # Treat skipped/unparseable as non-match for progress accounting.
                    false_matches += 1
                    batch_false += 1

                pair_index += 1
                if (
                    pair_index == 1
                    or pair_index == total_pairs
                    or (progress_every and pair_index % progress_every == 0)
                ):
                    progress_msg = (
                        f"    Labeled {pair_index}/{total_pairs if total_pairs else '?'}: "
                        f"{true_matches} matches, {false_matches} non-matches"
                    )
                    print(progress_msg)
                    progress_logger.info(
                        "LLM labeling: %d/%s total, matches=%d, non_matches=%d",
                        pair_index,
                        str(total_pairs) if total_pairs else "?",
                        true_matches,
                        false_matches,
                    )

            # Per-batch summary (helps when we chunk candidates into fixed-size batches).
            batch_secs = max(time.perf_counter() - batch_start, 0.0)
            progress_logger.info(
                "LLM batch %d/%d complete: batch_total=%d, batch_matches=%d, batch_non_matches=%d, "
                "cumulative_matches=%d, cumulative_non_matches=%d, batch_seconds=%.1f",
                batch_idx,
                batch_count if batch_count else batch_idx,
                int(len(batch)),
                batch_true,
                batch_false,
                true_matches,
                false_matches,
                batch_secs,
            )

        # Flush LLM logs and write artifacts
        if debug:
            self._write_debug_artifacts(out_dir, matches, pair_index)

            # Write configuration for this run
            config = {
                "model": getattr(chat_model, 'model_name', 'unknown'),
                "fields": fields,
                "few_shots_count": len(few_shots) if few_shots else 0,
                "generate_explanations": generate_explanations,
                "include_difficulty": include_difficulty,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retries": retries,
                "system_prompt_provided": bool(system_prompt),
                "parse_strictness": parse_strictness,
                "total_candidates": pair_index,
            }
            self._write_artifact("llm_config.json", config)

        # Always flush LLM logs (writes llm_calls.json and llm_usage_summary.json)
        self._llm_logger.flush(self._write_artifact)

        if matches:
            return pd.DataFrame(matches)
        else:
            columns = ["id1", "id2", "match"]
            if generate_explanations:
                columns.append("explanation")
            if include_difficulty:
                columns.append("difficulty")
            return pd.DataFrame(columns=columns)

    def _auto_select_fields(self, df_left: pd.DataFrame, df_right: pd.DataFrame, id_column: str, max_fields: int = 10) -> List[str]:
        """Auto-select string-like fields for LLM prompts."""
        # Get common columns (excluding id_column)
        common_cols = set(df_left.columns) & set(df_right.columns) - {id_column}

        # Prefer string-like columns
        string_cols = []
        for col in common_cols:
            if df_left[col].dtype == 'object' or df_right[col].dtype == 'object':
                string_cols.append(col)

        # If we have too many, limit to first few
        if len(string_cols) > max_fields:
            string_cols = string_cols[:max_fields]

        # If no string columns, take any common columns
        if not string_cols:
            string_cols = list(common_cols)[:max_fields]

        return string_cols

    def _build_prompt_template(
        self,
        system_prompt: Optional[str],
        few_shots: Optional[List[Tuple[Dict[str, Any], Dict[str, Any], str]]],
        generate_explanations: bool = True,
        include_difficulty: bool = False
    ) -> ChatPromptTemplate:
        """Build the chat prompt template with system message and optional few-shot examples."""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(generate_explanations, include_difficulty)

        messages = [SystemMessagePromptTemplate.from_template(system_prompt)]

        # Add few-shot examples if provided
        if few_shots:
            few_shot_examples = []
            for left_ex, right_ex, expected_json in few_shots:
                few_shot_examples.append({
                    "left_record": json.dumps(left_ex, ensure_ascii=False),
                    "right_record": json.dumps(right_ex, ensure_ascii=False),
                    "output": expected_json
                })

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=ChatPromptTemplate.from_messages([
                    ("human",
                     "Left record: {left_record}\nRight record: {right_record}"),
                    ("assistant", "{output}")
                ]),
                examples=few_shot_examples
            )
            messages.append(few_shot_prompt)

        # Add human message template for the actual comparison
        messages.append(HumanMessagePromptTemplate.from_template(
            "Left record: {left_record}\nRight record: {right_record}\n\n"
            "Return JSON matching the schema described above."
        ))

        return ChatPromptTemplate.from_messages(messages)

    def _get_default_system_prompt(self, generate_explanations: bool = True, include_difficulty: bool = False) -> str:
        """Get the default system prompt for entity matching."""
        # Build JSON format based on flags
        if generate_explanations and include_difficulty:
            json_format = '{{"match": true|false, "difficulty": "simple"|"medium"|"difficult", "explanation": "<brief explanation>"}}'
        elif generate_explanations:
            json_format = '{{"match": true|false, "explanation": "<brief explanation>"}}'
        elif include_difficulty:
            json_format = '{{"match": true|false, "difficulty": "simple"|"medium"|"difficult"}}'
        else:
            json_format = '{{"match": true|false}}'

        # Build guidelines based on flags
        guidelines = ["- match: true if records refer to the same entity, false otherwise"]

        if include_difficulty:
            guidelines.append("""- difficulty: how hard the decision was
  - "simple": obvious match or non-match (e.g., identical names, completely different entities)
  - "medium": requires some reasoning (e.g., abbreviations, minor variations)
  - "difficult": ambiguous case requiring careful analysis (e.g., similar but potentially different entities)""")

        if generate_explanations:
            guidelines.append("- explanation should be concise (1-2 sentences)")

        guidelines.append("- Consider variations in naming, formatting, abbreviations, and data quality")
        guidelines.append("- Respond with ONLY the JSON object and nothing else.")

        return f"""You are an expert entity resolver. Your task is to decide if two records refer to the same real-world entity.

Analyze the provided records carefully and return your decision as strict JSON in this format:
{json_format}

Guidelines:
{chr(10).join(guidelines)}"""

    def _serialize_record(self, record: pd.Series, fields: List[str], id_column: str, max_length: int = 200) -> str:
        """Serialize a record for the LLM prompt, including only specified fields."""
        data = {}

        for field in fields:
            if field in record and pd.notna(record[field]):
                value = str(record[field])
                # Truncate long strings
                if len(value) > max_length:
                    value = value[:max_length] + "..."
                data[field] = value

        return json.dumps(data, ensure_ascii=False)

    def _match_pair_with_retry(
        self,
        prompt_template: ChatPromptTemplate,
        left_data: str,
        right_data: str,
        chat_model: BaseChatModel,
        retries: int,
        temperature: Optional[float],
        max_tokens: Optional[int],
        pair_index: int,
        out_dir: str,
        debug: bool,
        parse_strictness: str,
        generate_explanations: bool,
        include_difficulty: bool,
        response_parser: Optional[Callable[[str],
                                           Optional[Dict[str, Any]]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Match a single pair with retry logic."""
        logger = logging.getLogger(__name__)

        for attempt in range(retries + 1):
            try:
                # Format the prompt
                messages = prompt_template.format_messages(
                    left_record=left_data,
                    right_record=right_data
                )

                # Log the comparison being made
                logger.info(f"[Pair {pair_index}] Comparing:")
                logger.info(f"  LEFT:  {left_data}")
                logger.info(f"  RIGHT: {right_data}")

                # Write debug artifacts
                if debug:
                    self._write_prompt_artifacts(
                        out_dir, pair_index, attempt, messages)

                # Call the model
                start_time = time.time()
                # Build kwargs without overriding model-level configuration
                invoke_kwargs: Dict[str, Any] = {}
                if max_tokens is not None:
                    invoke_kwargs["max_tokens"] = max_tokens
                # Do NOT pass temperature; rely on chat_model's own configuration

                response = chat_model.invoke(messages, **invoke_kwargs)
                duration = time.time() - start_time

                # Log the call using unified logger signature
                self._llm_logger.record_call(
                    chat_model=chat_model,
                    messages=messages,
                    response=response,
                    row_index=pair_index,
                    attempt=attempt,
                    duration_ms=duration * 1000.0,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                response_text = response.content

                # Log the LLM response
                logger.info(f"[Pair {pair_index}] LLM response: {response_text}")

                # Write debug artifacts
                if debug:
                    self._write_response_artifacts(
                        out_dir, pair_index, attempt, response_text)

                # Parse the response (custom parser first, then default)
                if response_parser is not None:
                    try:
                        custom = response_parser(response_text)
                        if isinstance(custom, dict):
                            # Normalize custom output
                            match = bool(custom.get("match", False))
                            result = {"match": match}
                            if generate_explanations:
                                result["explanation"] = str(custom.get("explanation", ""))
                            if include_difficulty:
                                result["difficulty"] = custom.get("difficulty", "")
                            return result
                    except Exception as e:
                        if parse_strictness == "skip":
                            pass  # fall through to default parser
                        else:
                            result = {"match": False}
                            if generate_explanations:
                                result["explanation"] = f"Custom parse error: {str(e)}"
                            if include_difficulty:
                                result["difficulty"] = ""
                            return result

                return self._parse_response(response_text, parse_strictness, generate_explanations, include_difficulty)

            except Exception as e:
                if debug:
                    self._write_error_artifacts(
                        out_dir, pair_index, attempt, str(e))

                if attempt < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # Final attempt failed
                    if parse_strictness == "zero_score":
                        return {
                            "score": 0.0,
                            "notes": f"LLM call failed after {retries + 1} attempts: {str(e)}"
                        }
                    else:
                        return None

        return None

    def _parse_response(self, response_text: str, parse_strictness: str, generate_explanations: bool, include_difficulty: bool) -> Optional[Dict[str, Any]]:
        """Parse the LLM response and extract match information."""
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json_from_response(response_text)
            if not json_str:
                raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Extract match decision
            match = bool(data.get("match", False))

            result = {"match": match}

            # Extract explanation if requested
            if generate_explanations:
                result["explanation"] = data.get("explanation", "")

            # Extract difficulty if requested
            if include_difficulty:
                result["difficulty"] = data.get("difficulty", "")

            return result

        except Exception as e:
            if parse_strictness == "skip":
                return None
            else:
                # Return non-match for parse errors
                result = {"match": False}
                if generate_explanations:
                    result["explanation"] = f"Parse error: {str(e)}"
                if include_difficulty:
                    result["difficulty"] = ""
                return result

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON object from response text, handling extra text."""
        # Look for JSON object boundaries
        start = response_text.find('{')
        if start == -1:
            return None

        # Find matching closing brace
        brace_count = 0
        for i, char in enumerate(response_text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response_text[start:i+1]

        return None

    def _write_prompt_artifacts(self, out_dir: str, pair_index: int, attempt: int, messages: List[BaseMessage]):
        """Write prompt artifacts for debugging."""
        prompts_dir = Path(out_dir) / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Write as text
        with open(prompts_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            for msg in messages:
                f.write(f"{msg.__class__.__name__}: {msg.content}\n\n")

        # Write as JSON
        with open(prompts_dir / f"row_{pair_index}_attempt_{attempt}.json", "w") as f:
            json.dump([{"type": msg.__class__.__name__, "content": msg.content}
                      for msg in messages], f, indent=2)

    def _write_response_artifacts(self, out_dir: str, pair_index: int, attempt: int, response_text: str):
        """Write response artifacts for debugging."""
        responses_dir = Path(out_dir) / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        with open(responses_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            f.write(response_text)

    def _write_error_artifacts(self, out_dir: str, pair_index: int, attempt: int, error_text: str):
        """Write error artifacts for debugging."""
        errors_dir = Path(out_dir) / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)

        with open(errors_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            f.write(error_text)

    def _write_debug_artifacts(self, out_dir: str, matches: List[Dict], total_candidates: int):
        """Write final debug artifacts."""
        # Write stats
        stats = {
            "total_candidates": total_candidates,
            "total_matches": len(matches),
            "match_rate": len(matches) / total_candidates if total_candidates > 0 else 0.0
        }

        with open(Path(out_dir) / "llm_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Write sample matches
        if matches:
            sample_df = pd.DataFrame(matches[:10])  # First 10 matches
            sample_df.to_csv(Path(out_dir) / "sample_matches.csv", index=False)

    def _write_artifact(self, artifact_name: str, content: Any):
        """Write artifact to current run directory.

        Supports JSON (dict/list), text (str), and DataFrame to CSV.
        """
        base_dir = self._current_run_out_dir or Path(
            "output/entitymatching/llm")
        filepath = base_dir / artifact_name
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(filepath, 'w') as f:
                json.dump(content, f, indent=2, default=str)
        elif isinstance(content, str):
            with open(filepath, 'w') as f:
                f.write(content)
        elif isinstance(content, pd.DataFrame):
            content.to_csv(filepath, index=False)
        else:
            # Fallback to string serialization
            with open(filepath, 'w') as f:
                f.write(str(content))
