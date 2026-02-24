"""
Add unit suffix to duration column in lastfm.csv and update target schema.
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("usecases/input/music/data")
SCHEMA_DIR = Path("usecases/input/music/schemamatching")


def convert_duration_column():
    """Add 'seconds' suffix to duration values in lastfm.csv."""
    lastfm_path = DATA_DIR / "lastfm.csv"
    df = pd.read_csv(lastfm_path)

    # Convert numeric duration to string with "seconds" suffix
    def add_unit(val):
        if pd.isna(val):
            return val
        return f"{int(val)} seconds"

    df["album_length_min"] = df["album_length_min"].apply(add_unit)

    # Save back
    df.to_csv(lastfm_path, index=False)
    print(f"Updated {lastfm_path}")
    print(f"Sample values: {df['album_length_min'].head(5).tolist()}")


def update_target_schema():
    """Add x-pydi-target-unit to duration field in target schema."""
    schema_path = SCHEMA_DIR / "target_schema.json"

    with open(schema_path) as f:
        schema = json.load(f)

    # Update duration field
    if "duration" in schema.get("properties", {}):
        schema["properties"]["duration"]["x-pydi-target-unit"] = "min"
        schema["properties"]["duration"]["description"] = "Total duration of the release in minutes"
        # Change type to number since we'll have decimal minutes
        schema["properties"]["duration"]["type"] = "number"
        schema["properties"]["duration"]["examples"] = [27.85, 13.75, 10.62]

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Updated {schema_path}")
    print(f"Duration field: {json.dumps(schema['properties']['duration'], indent=2)}")


if __name__ == "__main__":
    convert_duration_column()
    print()
    update_target_schema()
