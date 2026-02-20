#!/usr/bin/env python3
"""Debug script to see exactly what the LLM receives and returns for schema matching."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyDI.pipeline.run import load_data_file
from PyDI.pipeline.schema_matching import auto_match_schema
from langchain_openai import ChatOpenAI


def main():
    # Load discogs data
    data_path = Path("usecases/input/music/data/discogs.csv")
    schema_path = Path("usecases/input/music/schemamatching/target_schema.json")

    print(f"Loading data from {data_path}...")
    df = load_data_file(data_path)
    print(f"Columns: {list(df.columns)}")
    print()

    print(f"Loading schema from {schema_path}...")
    with open(schema_path) as f:
        target_schema = json.load(f)

    # Create model
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Run with debug mode
    output_dir = Path("scripts/output/schema_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning schema matching with debug mode...")
    print(f"Debug artifacts will be saved to: {output_dir}")
    print()

    mapping = auto_match_schema(
        df,
        target_schema,
        chat_model,
        num_rows=10,
        debug=True,
        out_dir=str(output_dir),
    )

    print("\n=== MAPPING RESULT ===")
    print(mapping.to_string())

    # Check if tracks was mapped
    if 'tracks_track-name' in mapping['source_column'].values:
        print("\n✓ tracks_track-name was mapped!")
    else:
        print("\n✗ tracks_track-name was NOT mapped!")
        print("\nCheck the debug artifacts in:", output_dir)
        print("- prompt_attempt_0.txt: The exact prompt sent to LLM")
        print("- response_attempt_0.txt: The LLM's raw response")


if __name__ == "__main__":
    main()
