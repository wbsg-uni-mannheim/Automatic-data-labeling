#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


seed_module = _load_module(
    "_run_seed_round_only_profiles_module",
    ROOT / "scripts" / "labeling" / "similarity_search.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue a crashed seed-round-only run in-place."
    )
    parser.add_argument(
        "run_dir",
        help="Existing run directory, e.g. output/seed_round_only_profiles/benchmark_abt-buy_20260415_190530",
    )
    parser.add_argument(
        "--config",
        default=str(seed_module.DEFAULT_CONFIG),
        help="Benchmark labeling config.",
    )
    parser.add_argument(
        "--benchmark",
        default="",
        help="Benchmark key. If omitted, infer it from the run directory.",
    )
    parser.add_argument(
        "--profiles",
        default="small,medium,large,all",
        help="Comma-separated profiles to materialize. Default: %(default)s",
    )
    parser.add_argument("--model", default="", help="Override labeling model from config.")
    parser.add_argument("--seed-max-calls", type=int, default=0, help="Override max seed-round calls.")
    parser.add_argument("--no-export-ditto", action="store_true", help="Skip Ditto train json.gz export.")
    parser.add_argument("--skip-random-profiles", action="store_true", help="Do not continue or build *_plus20random profiles.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the plan without continuing the run.")
    return parser.parse_args()


def _build_forwarded_argv(args: argparse.Namespace) -> List[str]:
    benchmark = str(args.benchmark).strip()
    if not benchmark:
        benchmark = seed_module._infer_benchmark_from_run_dir(Path(args.run_dir).expanduser().resolve()) or ""
    if not benchmark:
        raise ValueError("Could not infer benchmark from run directory. Please pass --benchmark explicitly.")

    argv: List[str] = [
        "similarity_search.py",
        "--config",
        str(args.config),
        "--benchmarks",
        benchmark,
        "--profiles",
        str(args.profiles),
        "--existing-run-dir",
        str(Path(args.run_dir).expanduser().resolve()),
    ]
    if str(args.model).strip():
        argv.extend(["--model", str(args.model).strip()])
    if int(args.seed_max_calls) > 0:
        argv.extend(["--seed-max-calls", str(int(args.seed_max_calls))])
    if args.no_export_ditto:
        argv.append("--no-export-ditto")
    if args.skip_random_profiles:
        argv.append("--skip-random-profiles")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def main() -> None:
    os.chdir(ROOT)
    args = _parse_args()
    sys.argv = _build_forwarded_argv(args)
    seed_module.main()


if __name__ == "__main__":
    main()
