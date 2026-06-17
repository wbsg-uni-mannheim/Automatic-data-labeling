# Server Fetch List

The final materialized training files referenced by the paper are expected to come from the canonical server-side experiment output. The run registry records paths rooted at:

```text
/work/aasteine/Automatic-data-labeling/
```

The core release requires the server directory:

```text
/work/aasteine/Automatic-data-labeling/output/labeling_cleaned/
```

Optional ablation and validation files are located in:

```text
/work/aasteine/Automatic-data-labeling/output/labeling_cleaned_llmvalid/
/work/aasteine/Automatic-data-labeling/output/labeling_cleaned_no_val/
/work/aasteine/Automatic-data-labeling/output/labeling_cleaned_size_scan/
```

## Copying Directly Into the Release Layout

From the repository root, copy the core files into the release layout:

```bash
mkdir -p paper_artifacts/training_sets/materialized/core
rsync -av /work/aasteine/Automatic-data-labeling/output/labeling_cleaned/ \
  paper_artifacts/training_sets/materialized/core/labeling_cleaned/
```

Optional files can be copied similarly:

```bash
mkdir -p paper_artifacts/training_sets/materialized/optional
rsync -av /work/aasteine/Automatic-data-labeling/output/labeling_cleaned_llmvalid/ \
  paper_artifacts/training_sets/materialized/optional/labeling_cleaned_llmvalid/
rsync -av /work/aasteine/Automatic-data-labeling/output/labeling_cleaned_no_val/ \
  paper_artifacts/training_sets/materialized/optional/labeling_cleaned_no_val/
rsync -av /work/aasteine/Automatic-data-labeling/output/labeling_cleaned_size_scan/ \
  paper_artifacts/training_sets/materialized/optional/labeling_cleaned_size_scan/
```

After copying, verify the core release:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier core --layout release
```

Verify all listed files:

```bash
python scripts/artifacts/verify_release_artifacts.py --tier all --layout release
```

## Known Local Gap

The local raw WDC active-learning file below was found to have an incomplete gzip stream and should be replaced from the server if raw, non-release intermediates are retained:

```text
training_data/three_phase_labeling_ditto_only_v2/benchmark_wdc_20260323_202820/profiles/all/active_labels_latest_wdc_all_train.json.gz
```
