# Stage1 evaluation

Run from the repository root:

```bash
/root/miniconda3/envs/neuloc_v0/bin/python -u experiments/eval_stage1/run.py --config experiments/eval_stage1/config.yaml
```

Model architecture, scene list and train/val/test split are inherited from the checkpoint's `opts.yaml`. Override evaluation settings in `config.yaml`; `scenes: []` evaluates every saved scene. Outputs use the checkpoint experiment name, epoch, evaluation name and timestamp, with incremental `metrics.json`, configuration snapshots and a completion marker.

- `retrieval_mode: north_aligned`: recorded query heading rotates the query north. Satellite crops stay north-aligned. Scale follows `val_gallery_scale_mode` and `val_query_scale2uniform`.
- `retrieval_mode: query_rotation_search`: each scene uses its mean scale for the gallery and the query. The query heading is not used to warp the image. The query is encoded every `query_rotation_step_deg` (default 10, must divide 360). Feature distance is `sqrt(max(0, 2 - 2 cos))`. Each angle contributes `exp(-distance / heading_temperature)` with `heading_temperature` default 0.02. Location score is the sum of those responses, and Recall@K ranks locations by that sum. At the winning location, the warp is the peak angle plus the softmax-weighted local deviation at the same temperature. The reported heading is the negation of that warp, in the `rotdeg_fm_north_anticlock` convention. Val and test each report mean and median absolute heading error in degrees.
- `reference_rotation_search` is reserved and raises `NotImplementedError`.
- `val_gallery_scale_mode: scene_mean`: one scene mean scale.
- `val_gallery_scale_mode: multiscale`: under `north_aligned`, use `val_gallery_scale_levels` bin centers across the scene scale bounds, spaced by `val_gallery_scale_spacing: linear | log`. Each position/scale pair is its own candidate. `query_rotation_search` rejects this mode.
- `val_query_rot2uniform`: `north_aligned` only. True uses the recorded heading to rotate north. False keeps the original query heading and does not build rotated candidates.
- `val_query_scale2uniform`: `north_aligned` only. True normalizes the query footprint to the scene mean. `query_rotation_search` always does this.
- `max_forward_variants`: maximum images in one network forward. Rotation and scale change a precomputed sampling grid; the original query is sampled directly and the sampled batch is capped at this size. `query_rotation_search` loads every query in a scene, expands all angles, then encodes those variants in chunks of this size. The default evaluation value is 512. When the key is omitted, forwards keep using `val_chunk_size`.

The same entrypoint supports 2DoF/4DoF trained encoders by changing `checkpoint` and evaluation settings. Training `pair_alignment_mode` does not automatically select the evaluation protocol. An omitted `retrieval_mode` stays `north_aligned`, including evaluation called during training. Recall's spatial tolerance remains the existing scene half-image radius. Val and test are reported separately.

Current baseline: `epoch010.pth`, `north_aligned`, scene mean gallery scale, query aligned north, query scale normalization disabled. Switch `retrieval_mode` to `query_rotation_search` to run the discrete query-rotation protocol.
