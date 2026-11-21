# Data preparation and maintenance

Training and inference commands are documented in the [experiment guides](../experiments/README.md). To evaluate the supplied models, follow the [paper reproduction workflow](../README.md#reproduce-inference-with-the-supplied-models).

The tools in this directory support dataset preparation and run maintenance. They are not required to regenerate the frozen evaluation splits supplied with the models.

## Export fixed splits for new experiments

From the repository root:

```bash
python scripts/export_data_splits.py --help
python scripts/export_data_splits.py \
  --manifest /path/to/ContinuScenes/dataset_manifest.json \
  --output-dir gen_fm_exps/data_preparation/splits
```

The output directory must not already exist. The exporter creates independent JSON files without modifying the source CSV files. It generates quality, quality-plus-edge, and quality-plus-edge-plus-scale profiles, each with segment and interval assignments. Inspect the exported files before using them as a new dataset protocol.

For evaluation with the paper models, use the bundle's frozen assignments instead of newly generated splits.

`update_data_splits.py` updates split columns in CSV files for workflows that use CSV-based assignments. Checkpoint and log cleanup utilities are also provided in this directory; review their arguments before using them on saved runs.
