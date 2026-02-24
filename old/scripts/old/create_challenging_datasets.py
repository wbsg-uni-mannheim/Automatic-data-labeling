"""
Create Challenging Schema Matching Test Datasets

This script creates modified versions of the existing use cases with:
1. Challenging headers - column names that are less similar to target schema
2. No headers - data without column headers (for testing headerless schema matching)

Usage:
    python create_challenging_datasets.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from PyDI.io import load_xml

BASE_DIR = Path(__file__).parent.parent / "usecases" / "input"


# =============================================================================
# Header Mapping Definitions
# =============================================================================

# Music use case - make headers less similar to target schema
# Target schema columns: id, name, artist, release-date, release-country, label, genre, tracks
MUSIC_CHALLENGING_HEADERS = {
    # discogs.xml columns
    "discogs": {
        "id": "rec_uid",           # obscure: "record unique identifier"
        "name": "title_str",       # different naming: not "name"
        "artist": "performer",     # synonym: different word
        "release-date": "pub_dt",  # abbreviation: "publication date"
        "release-country": "origin_loc",  # abstract: "origin location"
        "label": "imprint",        # industry term
        "genre": "category",       # generic term
        "tracks": "tracklist",     # slightly different
    },
    # lastfm.xml columns
    "lastfm": {
        "id": "item_code",         # generic identifier
        "name": "album_title",     # specific but different
        "artist": "band",          # alternative term
        "tracks": "song_list",     # different naming
    },
    # musicbrainz.xml columns
    "musicbrainz": {
        "id": "mb_ref",            # abbreviation
        "name": "release_title",   # different naming
        "artist": "creator",       # abstract term
        "release-date": "issued",  # different verb
        "release-country": "territory",  # geographic term
        "duration": "length_sec",  # with unit hint
        "tracks": "compositions",  # abstract
    },
}

# Games use case - make headers less similar to target schema
# Target schema: id, name, releaseYear, developer, publisher, genres, platform, criticScore, userScore, ESRB, globalSales, series
GAMES_CHALLENGING_HEADERS = {
    # dbpedia.xml columns
    "dbpedia": {
        "id": "wiki_ref",          # source-specific
        "name": "title",           # generic
        "releaseYear": "launch_yr",  # different term
        "developer": "studio",     # industry term
        "genres": "classification",  # abstract
        "platform": "system",      # alternative term
        "series": "franchise",     # industry term
    },
    # metacritic.xml columns
    "metacritic": {
        "id": "mc_id",             # source abbreviation
        "name": "game_title",      # specific
        "releaseYear": "year_published",  # verbose
        "developer": "made_by",    # casual phrasing
        "genres": "type",          # generic
        "platform": "console",     # specific term
        "criticScore": "press_rating",  # different naming
        "userScore": "player_rating",  # different naming
        "ESRB": "age_rating",      # descriptive
    },
    # sales.xml columns
    "sales": {
        "id": "sku",               # retail term
        "name": "product_name",    # retail term
        "releaseYear": "release_period",  # vague
        "developer": "creator_studio",  # combined
        "genres": "game_type",     # specific
        "publisher": "distributor",  # related but different
        "platform": "hw_platform",  # abbreviated
        "criticScore": "review_avg",  # different concept
        "userScore": "community_score",  # different naming
        "ESRB": "content_rating",  # descriptive
        "globalSales": "units_sold",  # different metric
    },
}

# Companies use case - make headers less similar to target schema
# Target schema: id, name, website, founded, country, city, industry, assets, revenue, keypeople/founders
COMPANIES_CHALLENGING_HEADERS = {
    # dbpedia.xml columns
    "dbpedia": {
        "id": "entity_uri",        # technical term
        "name": "org_name",        # abbreviated
        "website": "homepage",     # alternative
        "founded": "established",  # synonym
        "country": "nation",       # synonym
        "city": "headquarters",    # related concept
        "industry": "sector",      # business term
        "assets": "total_assets_val",  # verbose
        "revenue": "annual_income",  # different metric name
        "keypeople": "executives",  # different term
    },
    # forbes.xml columns
    "forbes": {
        "id": "forbes_url",        # source-specific
        "name": "company",         # generic
        "website": "url",          # abbreviated
        "country": "region",       # geographic
        "industry": "business_segment",  # verbose
        "assets": "asset_value",   # verbose
        "revenue": "sales_figure",  # different term
    },
    # fullcontact.xml columns
    "fullcontact": {
        "id": "fc_id",             # source abbreviation
        "name": "organization",    # formal
        "founded": "year_started",  # casual
        "country": "location_country",  # verbose
        "city": "location_city",   # verbose
        "keypeople": "leadership",  # different term
    },
}


def load_and_transform_xml(xml_path: Path, header_mapping: dict, dataset_name: str) -> pd.DataFrame:
    """Load XML file and apply header transformations."""
    # Load XML
    df = load_xml(
        xml_path,
        name=dataset_name,
        nested_handling="aggregate",
        add_index=True,
        index_column_name="id",
        id_prefix=dataset_name,
    )

    # Apply header mapping
    rename_map = {}
    for old_col, new_col in header_mapping.items():
        if old_col in df.columns:
            rename_map[old_col] = new_col

    df = df.rename(columns=rename_map)

    return df


def create_no_header_csv(df: pd.DataFrame, output_path: Path):
    """Save DataFrame as CSV without headers."""
    df.to_csv(output_path, index=False, header=False)


def create_with_header_csv(df: pd.DataFrame, output_path: Path):
    """Save DataFrame as CSV with headers."""
    df.to_csv(output_path, index=False)


def process_music():
    """Process music use case."""
    print("\n=== Processing Music Use Case ===")

    input_dir = BASE_DIR / "music" / "data"
    challenging_dir = BASE_DIR / "music_challenging" / "data"
    no_header_dir = BASE_DIR / "music_no_headers" / "data"

    for xml_file in input_dir.glob("*.xml"):
        dataset_name = xml_file.stem
        print(f"  Processing {dataset_name}...")

        # Get header mapping for this dataset
        header_mapping = MUSIC_CHALLENGING_HEADERS.get(dataset_name, {})

        # Load and transform
        df = load_and_transform_xml(xml_file, header_mapping, dataset_name)

        # Save challenging version
        csv_path = challenging_dir / f"{dataset_name}.csv"
        create_with_header_csv(df, csv_path)
        print(f"    Created {csv_path} with columns: {list(df.columns)}")

        # For no-header version, use original column order but no headers
        df_original = load_xml(
            xml_file,
            name=dataset_name,
            nested_handling="aggregate",
            add_index=True,
            index_column_name="id",
            id_prefix=dataset_name,
        )
        no_header_path = no_header_dir / f"{dataset_name}.csv"
        create_no_header_csv(df_original, no_header_path)
        print(f"    Created {no_header_path} (no headers)")


def process_games():
    """Process games use case."""
    print("\n=== Processing Games Use Case ===")

    input_dir = BASE_DIR / "games" / "data"
    challenging_dir = BASE_DIR / "games_challenging" / "data"
    no_header_dir = BASE_DIR / "games_no_headers" / "data"

    for xml_file in input_dir.glob("*.xml"):
        dataset_name = xml_file.stem
        print(f"  Processing {dataset_name}...")

        # Get header mapping for this dataset
        header_mapping = GAMES_CHALLENGING_HEADERS.get(dataset_name, {})

        # Load and transform
        df = load_and_transform_xml(xml_file, header_mapping, dataset_name)

        # Save challenging version
        csv_path = challenging_dir / f"{dataset_name}.csv"
        create_with_header_csv(df, csv_path)
        print(f"    Created {csv_path} with columns: {list(df.columns)}")

        # For no-header version, use original column order but no headers
        df_original = load_xml(
            xml_file,
            name=dataset_name,
            nested_handling="aggregate",
            add_index=True,
            index_column_name="id",
            id_prefix=dataset_name,
        )
        no_header_path = no_header_dir / f"{dataset_name}.csv"
        create_no_header_csv(df_original, no_header_path)
        print(f"    Created {no_header_path} (no headers)")


def process_companies():
    """Process companies use case."""
    print("\n=== Processing Companies Use Case ===")

    input_dir = BASE_DIR / "companies" / "data"
    challenging_dir = BASE_DIR / "companies_challenging" / "data"
    no_header_dir = BASE_DIR / "companies_no_headers" / "data"

    for xml_file in input_dir.glob("*.xml"):
        dataset_name = xml_file.stem
        print(f"  Processing {dataset_name}...")

        # Get header mapping for this dataset
        header_mapping = COMPANIES_CHALLENGING_HEADERS.get(dataset_name, {})

        # Load and transform
        df = load_and_transform_xml(xml_file, header_mapping, dataset_name)

        # Save challenging version
        csv_path = challenging_dir / f"{dataset_name}.csv"
        create_with_header_csv(df, csv_path)
        print(f"    Created {csv_path} with columns: {list(df.columns)}")

        # For no-header version, use original column order but no headers
        df_original = load_xml(
            xml_file,
            name=dataset_name,
            nested_handling="aggregate",
            add_index=True,
            index_column_name="id",
            id_prefix=dataset_name,
        )
        no_header_path = no_header_dir / f"{dataset_name}.csv"
        create_no_header_csv(df_original, no_header_path)
        print(f"    Created {no_header_path} (no headers)")


def main():
    print("=" * 70)
    print("Creating Challenging Schema Matching Test Datasets")
    print("=" * 70)

    process_music()
    process_games()
    process_companies()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nCreated directories:")
    print("  - usecases/input/music_challenging/data")
    print("  - usecases/input/music_no_headers/data")
    print("  - usecases/input/games_challenging/data")
    print("  - usecases/input/games_no_headers/data")
    print("  - usecases/input/companies_challenging/data")
    print("  - usecases/input/companies_no_headers/data")
    print("\nTest with:")
    print("  python test_schema_matching.py \\")
    print("      --data-dir usecases/input/music_challenging/data \\")
    print("      --schema usecases/input/music/schemamatching/target_schema.json \\")
    print("      --output-dir scripts/output/schema_test/music_challenging")


if __name__ == "__main__":
    main()
