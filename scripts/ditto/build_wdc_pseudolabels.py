#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.ditto_modern.pseudolabels import build_pseudolabels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WDC pseudo-labeled json.gz from multi-agent predictions")
    parser.add_argument("--source-json-gz", required=True)
    parser.add_argument("--multi-agent-run-dir", required=True)
    parser.add_argument("--output-json-gz", required=True)
    parser.add_argument("--policy", choices=["consensus", "majority"], default="consensus")
    parser.add_argument("--include-majority-with-weight", action="store_true")
    parser.add_argument("--majority-weight", type=float, default=0.5)
    args = parser.parse_args()

    report = build_pseudolabels(
        source_json_gz=args.source_json_gz,
        multi_agent_run_dir=args.multi_agent_run_dir,
        output_json_gz=args.output_json_gz,
        policy=args.policy,
        include_majority_with_weight=args.include_majority_with_weight,
        majority_weight=args.majority_weight,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
