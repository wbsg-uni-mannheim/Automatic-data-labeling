# Human Audit Materials

The human audit materials are retained in their original output locations. They document the test-label audit described in the paper.

## Sampling Frame and Review Interface

- `analysis/test_teacher_audit_20260528/test_teacher_audit_master.csv`
- `analysis/test_teacher_audit_20260528/summary.json`
- `analysis/test_teacher_audit_20260528/batch_*.csv`
- `analysis/test_teacher_audit_20260528/batch_*.html`

The master CSV and batch exports define the reviewed pairs and the audit batches. The HTML files are review-interface exports and should be treated as documentation of the annotation environment rather than as the authoritative human decision table.

## Human Decisions

- `results/error_anlysis/labelers/*_annotations.csv`
- `results/error_anlysis/labeler_error_rates.csv`

The `human_decision` column in the annotation files records the final human audit decision. The spelling of `error_anlysis` is preserved because it is the historical experiment-output directory referenced by existing scripts and result files.
