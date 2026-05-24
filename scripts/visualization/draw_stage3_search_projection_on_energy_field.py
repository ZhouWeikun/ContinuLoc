#!/usr/bin/env python3
"""Draw Stage-3 4D search candidates projected onto a 2D energy field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def _maybe_reexec_with_env_lib() -> None:
    """Put the active conda env lib first before importing pandas/torch dependents."""
    if os.environ.get("_STAGE3_SEARCH_PROJECTION_BOOTSTRAPPED") == "1":
        return

    python_bin = os.path.abspath(sys.executable)
    if not python_bin.endswith("/bin/python"):
        return

    env_root = os.path.dirname(os.path.dirname(python_bin))
    env_lib = os.path.join(env_root, "lib")
    if not os.path.isdir(env_lib):
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld.split(":") if part]
    if ld_parts and ld_parts[0] == env_lib:
        return

    env = os.environ.copy()
    env["_STAGE3_SEARCH_PROJECTION_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else env_lib
    os.execve(python_bin, [python_bin] + sys.argv, env)


_maybe_reexec_with_env_lib()

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.stage3_ingp_inferencer import Stage3INGPInferencer  # noqa: E402
from trainers.util_stage3_energy_field_analyzer import Stage3EnergyFieldAnalyzer  # noqa: E402


STAGE_SPECS = {
    "grid": {
        "coords_key": "coords_grid",
        "scores_key": "scores_grid",
        "label": "Stage I",
        "color": "#29a3ff",
        "marker": "o",
    },
    "mode": {
        "coords_key": "coords_mode",
        "scores_key": "scores_mode",
        "label": "Stage II",
        "color": "#ffb000",
        "marker": "s",
    },
    "evo": {
        "coords_key": "coords_evo",
        "scores_key": "scores_evo",
        "label": "Stage III",
        "color": "#19c37d",
        "marker": "^",
    },
}

TOP1_LABEL_FIELDS = {"stage", "row", "col", "r", "s", "score", "none"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_bundle(run_dir: Path) -> dict[str, Any]:
    bundle_path = run_dir / "stage3_retrieval_bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing stage3 retrieval bundle: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected bundle dict from {bundle_path}, got {type(bundle).__name__}.")
    required = {"coords_grid", "scores_grid", "coords_mode", "scores_mode", "coords_evo", "scores_evo", "coords_gt"}
    missing = sorted(required - set(bundle.keys()))
    if missing:
        raise KeyError(f"Bundle missing required keys: {', '.join(missing)}")
    return bundle


def _load_source_indices(run_dir: Path, n_bundle: int) -> list[int] | None:
    """Read dataset source indices exported with stage3 triplets, if available."""
    summary_csv = run_dir / "stage3_triplets_summary.csv"
    if not summary_csv.is_file():
        return None

    source_indices: list[int | None] = [None] * int(n_bundle)
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "query_id" not in (reader.fieldnames or []) or "source_index" not in (reader.fieldnames or []):
            return None
        for row in reader:
            row_idx = int(row["query_id"])
            if 0 <= row_idx < int(n_bundle):
                source_indices[row_idx] = int(row["source_index"])

    missing = [idx for idx, value in enumerate(source_indices) if value is None]
    if missing:
        raise ValueError(
            f"{summary_csv} does not provide source_index for all bundle rows; "
            f"first missing rows: {missing[:10]}"
        )
    return [int(value) for value in source_indices]


def _load_run_row_meta(run_dir: Path) -> dict[int, dict[str, Any]]:
    summary_csv = run_dir / "stage3_triplets_summary.csv"
    if not summary_csv.is_file():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "query_id" not in (reader.fieldnames or []):
            return {}
        for row in reader:
            rows[int(row["query_id"])] = dict(row)
    return rows


def _build_inferencer(p_yaml: Path, stage3_args: list[str]) -> Stage3INGPInferencer:
    argv = [sys.argv[0], "--p_yaml", str(p_yaml), *stage3_args]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        inferencer = Stage3INGPInferencer()
    finally:
        sys.argv = old_argv
    return inferencer


def _init_runtime(inferencer: Stage3INGPInferencer, use_train_uav: bool) -> None:
    inferencer._init_stage3_test_runtime(use_train_uav=use_train_uav)
    inferencer._load_checkpoints_for_inference()
    inferencer._enter_inference_eval_mode()


def _normalize_energy_field_local_zoom_option(inferencer: Stage3INGPInferencer) -> None:
    local_zoom_wh = getattr(inferencer.opt, "stage3_energy_field_local_zoom_wh", None)
    if isinstance(local_zoom_wh, str) and local_zoom_wh.strip().lower() in {"", "none", "null"}:
        setattr(inferencer.opt, "stage3_energy_field_local_zoom_wh", None)


def _get_split_coords(inferencer: Stage3INGPInferencer, query_id: int, use_train_uav: bool) -> torch.Tensor | None:
    dataset = inferencer.uav_dataset_train if use_train_uav else inferencer.uav_dataset_test
    if dataset is None:
        return None
    coords_all = getattr(
        dataset,
        "uav_coords_4d_torch_train" if use_train_uav else "uav_coords_4d_torch_test",
        None,
    )
    if coords_all is None:
        return None
    query_id = int(query_id)
    if query_id < 0 or query_id >= int(coords_all.shape[0]):
        return None
    return torch.as_tensor(coords_all[query_id], dtype=torch.float32).detach().cpu()


def _coord_max_abs_delta(coord_a: torch.Tensor | None, coord_b: torch.Tensor) -> float | None:
    if coord_a is None:
        return None
    coord_a = torch.as_tensor(coord_a, dtype=torch.float32).detach().cpu().reshape(-1)
    coord_b = torch.as_tensor(coord_b, dtype=torch.float32).detach().cpu().reshape(-1)
    if coord_a.numel() != coord_b.numel():
        return None
    delta = torch.abs(coord_a - coord_b)
    if delta.numel() >= 3:
        rot_delta = torch.remainder(delta[2] + math.pi, 2 * math.pi) - math.pi
        delta[2] = torch.abs(rot_delta)
    return float(delta.max().item())


def _resolve_energy_query_id(
    inferencer: Stage3INGPInferencer,
    bundle: dict[str, Any],
    row_idx: int,
    source_indices: list[int] | None,
    use_train_uav: bool,
    coord_tol: float = 1e-5,
) -> tuple[int, dict[str, Any]]:
    gt_coord = torch.as_tensor(bundle["coords_gt"][row_idx], dtype=torch.float32).detach().cpu()
    row_coord_delta = _coord_max_abs_delta(
        _get_split_coords(inferencer, row_idx, use_train_uav=use_train_uav),
        gt_coord,
    )
    source_index = None if source_indices is None else int(source_indices[row_idx])
    source_coord_delta = None
    if source_index is not None:
        source_coord_delta = _coord_max_abs_delta(
            _get_split_coords(inferencer, source_index, use_train_uav=use_train_uav),
            gt_coord,
        )

    if row_coord_delta is not None and row_coord_delta <= float(coord_tol):
        energy_query_id = int(row_idx)
        source = "bundle_row_id"
    elif source_coord_delta is not None and source_coord_delta <= float(coord_tol):
        energy_query_id = int(source_index)
        source = "stage3_triplets_summary.csv/source_index"
    elif row_coord_delta is not None:
        energy_query_id = int(row_idx)
        source = "bundle_row_id_coord_mismatch_fallback"
    else:
        energy_query_id = int(row_idx)
        source = "bundle_row_id_no_coord_check"

    return energy_query_id, {
        "source": source,
        "source_index": source_index,
        "row_coord_delta_max_abs": row_coord_delta,
        "source_coord_delta_max_abs": source_coord_delta,
        "coord_tolerance": float(coord_tol),
    }


def _resolve_energy_job(inferencer: Stage3INGPInferencer, query_id: int, use_train_uav: bool, render_background: bool) -> dict[str, Any]:
    jobs = inferencer._build_energy_field_jobs(
        use_train_uav=use_train_uav,
        query_ids=[int(query_id)],
        use_sat_background_default=render_background,
    )
    if len(jobs) != 1:
        raise RuntimeError(f"Expected one energy-field job for query_id={query_id}, got {len(jobs)}.")
    return jobs[0]


def _compute_energy_result(
    analyzer: Stage3EnergyFieldAnalyzer,
    job: dict[str, Any],
    use_train_uav: bool,
) -> dict[str, Any]:
    return analyzer.analyze(
        n_nr=int(job["n_nr"]),
        n_nc=int(job["n_nc"]),
        use_train_uav=use_train_uav,
        query_id=int(job["query_id"]),
        plot_mode="ingp",
        local_zoom_wh=job["local_zoom_wh"],
        vis=False,
        render_map_contour=False,
        ref_feat_backend=job["ref_feat_backend"],
        chunk_size_vis=int(job["chunk_size_vis"]),
        analyse_fft=False,
    )


def _resolve_energy_common_config(inferencer: Stage3INGPInferencer, use_train_uav: bool, render_background: bool) -> dict[str, Any]:
    cfg = inferencer._resolve_energy_field_common_config(
        use_train_uav=use_train_uav,
        query_ids=[0],
        use_sat_background_default=render_background,
    )
    return cfg


def _bounds_from_coord(inferencer: Stage3INGPInferencer, coord: torch.Tensor, local_zoom_wh: Any) -> dict[str, float]:
    sat_dataset = inferencer.sat_dataset
    global_nr_min = float(sat_dataset.nr2sample_min)
    global_nr_max = float(sat_dataset.nr2sample_max)
    global_nc_min = float(sat_dataset.nc2sample_min)
    global_nc_max = float(sat_dataset.nc2sample_max)
    if local_zoom_wh is None:
        return {
            "start_nr": global_nr_min,
            "end_nr": global_nr_max,
            "start_nc": global_nc_min,
            "end_nc": global_nc_max,
        }

    zoom_nr, zoom_nc = local_zoom_wh
    half_nr = (global_nr_max - global_nr_min) * float(zoom_nr) / 2.0
    half_nc = (global_nc_max - global_nc_min) * float(zoom_nc) / 2.0
    center_nr = float(coord[0].item())
    center_nc = float(coord[1].item())
    return {
        "start_nr": max(global_nr_min, center_nr - half_nr),
        "end_nr": min(global_nr_max, center_nr + half_nr),
        "start_nc": max(global_nc_min, center_nc - half_nc),
        "end_nc": min(global_nc_max, center_nc + half_nc),
    }


def _prepare_sat_background_from_bounds(inferencer: Stage3INGPInferencer, bounds: dict[str, float]) -> Any:
    sat_dataset = inferencer.sat_dataset
    satimg_id = int(getattr(inferencer.opt, "stage3_energy_field_satimg_id", 0))
    try:
        sat_patch = sat_dataset.crop_rect_satimg(
            nrc_topleft=(bounds["start_nr"], bounds["start_nc"]),
            nrc_rightbottom=(bounds["end_nr"], bounds["end_nc"]),
            type="tensor",
            satimg_id=satimg_id,
        )
    except TypeError:
        sat_patch = sat_dataset.crop_rect_satimg(
            nrc_topleft=(bounds["start_nr"], bounds["start_nc"]),
            nrc_rightbottom=(bounds["end_nr"], bounds["end_nc"]),
            type="tensor",
        )
    return sat_dataset.denormalize_img(sat_patch)


def _resolve_energy_grid_shape_from_bounds(inferencer: Stage3INGPInferencer, bounds: dict[str, float]) -> tuple[int, int]:
    n_nr = getattr(inferencer.opt, "stage3_energy_field_n_nr", None)
    n_nc = getattr(inferencer.opt, "stage3_energy_field_n_nc", None)
    if n_nr is not None and n_nc is not None:
        return int(n_nr), int(n_nc)
    return inferencer._estimate_energy_field_grid_from_bounds_overlap(
        span_nr=float(bounds["end_nr"]) - float(bounds["start_nr"]),
        span_nc=float(bounds["end_nc"]) - float(bounds["start_nc"]),
        overlap=getattr(inferencer.opt, "stage3_field_overlap", 0.875),
    )


def _load_query_tensor_for_row(
    inferencer: Stage3INGPInferencer,
    row_idx: int,
    row_meta: dict[str, Any] | None,
    use_train_uav: bool,
) -> tuple[torch.Tensor, str | None]:
    dataset = inferencer.uav_dataset_train if use_train_uav else inferencer.uav_dataset_test
    query_path = None if row_meta is None else row_meta.get("query_path")
    if query_path:
        from PIL import Image

        transform = getattr(dataset, "uav_transform_test", None)
        if transform is None:
            transform = getattr(dataset, "uav_transform_train", None)
        if transform is None:
            raise AttributeError("UAV dataset does not expose uav_transform_test or uav_transform_train.")
        with Image.open(query_path) as img:
            query_tensor = transform(img.convert("RGB"))
        return query_tensor, str(query_path)

    query_tensor, _ = dataset[int(row_idx)]
    return query_tensor, None


def _compute_energy_result_from_bundle_row(
    analyzer: Stage3EnergyFieldAnalyzer,
    inferencer: Stage3INGPInferencer,
    bundle: dict[str, Any],
    row_idx: int,
    row_meta: dict[str, Any] | None,
    use_train_uav: bool,
    render_background: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _resolve_energy_common_config(
        inferencer=inferencer,
        use_train_uav=use_train_uav,
        render_background=render_background,
    )
    gt_coord = torch.as_tensor(bundle["coords_gt"][row_idx], dtype=torch.float32).detach().cpu()
    bounds = _bounds_from_coord(inferencer, gt_coord, local_zoom_wh=cfg["local_zoom_wh"])
    n_nr, n_nc = _resolve_energy_grid_shape_from_bounds(inferencer, bounds)
    query_tensor, query_path = _load_query_tensor_for_row(
        inferencer=inferencer,
        row_idx=row_idx,
        row_meta=row_meta,
        use_train_uav=use_train_uav,
    )
    imgs = query_tensor.unsqueeze(0).to(inferencer.device)
    coord_q = gt_coord.to(inferencer.device)
    query = {
        "query_id": int(row_idx),
        "image": imgs,
        "coord_q": coord_q,
        "feat_vis": inferencer._get_feats_fm_imgs(imgs),
        "set_name": "train" if use_train_uav else "test",
        "query_path": query_path,
    }
    grid = analyzer.build_grid(coord_q, n_nr=n_nr, n_nc=n_nc, local_zoom_wh=cfg["local_zoom_wh"], verbose=False)
    fields = analyzer.compute_fields(
        query=query,
        grid=grid,
        plot_mode="ingp",
        ref_feat_backend=cfg["ref_feat_backend"],
        chunk_size_vis=int(cfg["chunk_size_vis"]),
    )
    result = {
        "query": query,
        "grid": grid,
        "fields": fields,
        "plot_mode": "ingp",
        "ref_feat_backend": cfg["ref_feat_backend"],
        "saved_paths": [],
    }
    job = {
        **cfg,
        "query_id": int(row_idx),
        "energy_coord_source": "bundle.coords_gt",
        "query_image_source": query_path or "dataset_split_row",
        "sat_patch_np": _prepare_sat_background_from_bounds(inferencer, bounds) if render_background else None,
        "n_nr": int(n_nr),
        "n_nc": int(n_nc),
    }
    return result, job


def _stage_coords_scores(bundle: dict[str, Any], stage: str, row_idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    spec = STAGE_SPECS[stage]
    coords = torch.as_tensor(bundle[spec["coords_key"]][row_idx], dtype=torch.float32).detach().cpu()
    if coords.ndim == 1:
        coords = coords.view(1, -1)
    if coords.ndim != 2 or coords.shape[-1] != 4:
        raise ValueError(f"{spec['coords_key']}[{row_idx}] must have shape [K,4], got {tuple(coords.shape)}.")

    scores = bundle.get(spec["scores_key"])
    if scores is None:
        return coords, None
    scores_t = torch.as_tensor(scores[row_idx], dtype=torch.float32).detach().cpu()
    if scores_t.ndim == 0:
        scores_t = scores_t.view(1)
    return coords, scores_t


def _finite_stage_candidates(
    bundle: dict[str, Any],
    stage: str,
    row_idx: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    coords, scores = _stage_coords_scores(bundle, stage, row_idx)
    finite_mask = torch.isfinite(coords).all(dim=-1)
    coords = coords[finite_mask]
    if scores is not None:
        scores = scores[finite_mask[: scores.shape[0]]] if scores.shape[0] == finite_mask.shape[0] else scores[: coords.shape[0]]
    k = min(max(int(top_k), 1), int(coords.shape[0]))
    return coords[:k].numpy(), None if scores is None else scores[:k].numpy()


def _candidate_window_stats(coords: np.ndarray, raw_bounds: dict[str, float]) -> dict[str, int]:
    if coords.size == 0:
        return {"total": 0, "inside": 0, "outside": 0}
    nr = coords[:, 0]
    nc = coords[:, 1]
    inside = (
        (nr >= float(raw_bounds["start_nr"]))
        & (nr <= float(raw_bounds["end_nr"]))
        & (nc >= float(raw_bounds["start_nc"]))
        & (nc <= float(raw_bounds["end_nc"]))
    )
    return {"total": int(coords.shape[0]), "inside": int(inside.sum()), "outside": int((~inside).sum())}


def _pose_arrow_delta(rot_rad: float, arrow_len: float) -> tuple[float, float]:
    return math.sin(float(rot_rad)) * arrow_len, -math.cos(float(rot_rad)) * arrow_len


def _format_zoom_value_for_filename(value: float) -> str:
    text = f"{float(value):.6g}"
    if text.startswith("0."):
        return "0" + text[2:]
    if text.startswith("-0."):
        return "m0" + text[3:]
    return text.replace(".", "p").replace("-", "m")


def _format_zoom_suffix_for_filename(local_zoom_wh: Any) -> str:
    if local_zoom_wh is None:
        return "global"
    if len(local_zoom_wh) != 2:
        raise ValueError(f"local_zoom_wh must be None or length-2, got {local_zoom_wh!r}.")
    h_tag = _format_zoom_value_for_filename(float(local_zoom_wh[0]))
    w_tag = _format_zoom_value_for_filename(float(local_zoom_wh[1]))
    return f"h{h_tag}w{w_tag}"


def _parse_top1_label_fields(value: str) -> list[str]:
    fields = [field.strip().lower() for field in str(value).split(",") if field.strip()]
    if not fields:
        raise ValueError("--top1-label-fields must contain at least one field or 'none'.")
    unknown = sorted(set(fields) - TOP1_LABEL_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown --top1-label-fields values: {', '.join(unknown)}. "
            f"Allowed values: {', '.join(sorted(TOP1_LABEL_FIELDS))}."
        )
    if "none" in fields and len(fields) > 1:
        raise ValueError("--top1-label-fields 'none' cannot be combined with other fields.")
    return fields


def _format_top1_label(
    *,
    fields: list[str],
    stage_label: str,
    coord: np.ndarray,
    score: float | None,
) -> str | None:
    if fields == ["none"]:
        return None
    parts: list[str] = []
    for field in fields:
        if field == "stage":
            parts.append(stage_label)
        elif field == "row":
            parts.append(f"row={float(coord[0]):.2f}")
        elif field == "col":
            parts.append(f"col={float(coord[1]):.2f}")
        elif field == "r":
            parts.append(f"r={math.degrees(float(coord[2])):.1f}")
        elif field == "s":
            parts.append(f"s={float(coord[3]):.3g}")
        elif field == "score":
            parts.append("score=nan" if score is None else f"score={float(score):.3g}")
    return " ".join(parts)


def _resolve_candidate_top_k_by_stage(args: argparse.Namespace) -> dict[str, int]:
    global_top_k = int(args.candidate_top_k)
    if global_top_k <= 0:
        raise ValueError("--candidate-top-k must be positive.")

    top_k_by_stage = {
        "grid": global_top_k if args.candidate_top_k_grid is None else int(args.candidate_top_k_grid),
        "mode": global_top_k if args.candidate_top_k_mode is None else int(args.candidate_top_k_mode),
        "evo": global_top_k if args.candidate_top_k_evo is None else int(args.candidate_top_k_evo),
    }
    bad = {stage: top_k for stage, top_k in top_k_by_stage.items() if top_k <= 0}
    if bad:
        details = ", ".join(f"{stage}={top_k}" for stage, top_k in bad.items())
        raise ValueError(f"candidate top-k values must be positive: {details}")
    return top_k_by_stage


def _build_map_plot_setting(args: argparse.Namespace) -> dict[str, Any]:
    setting = {
        "draw_heatmap": bool(args.draw_heatmap),
        "draw_contour": bool(args.draw_contour),
        "cmap": str(args.cmap),
        "heatmap_alpha": float(args.heatmap_alpha),
        "n_fill_levels": int(args.n_fill_levels),
        "n_line_levels": int(args.n_line_levels),
        "contour_alpha": float(args.contour_alpha),
        "contour_color": str(args.contour_color),
        "contour_line_width": float(args.contour_line_width),
        "background_alpha": float(args.background_alpha),
    }
    for name in ("heatmap_alpha", "contour_alpha", "background_alpha"):
        if not 0.0 <= float(setting[name]) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if int(setting["n_fill_levels"]) < 2:
        raise ValueError("--n-fill-levels must be >= 2.")
    if int(setting["n_line_levels"]) < 2:
        raise ValueError("--n-line-levels must be >= 2.")
    if float(setting["contour_line_width"]) <= 0:
        raise ValueError("--contour-line-width must be positive.")
    return setting


def _render_projection(
    *,
    result: dict[str, Any],
    bundle: dict[str, Any],
    row_idx: int,
    query_id: int,
    energy_query_id: int,
    output_path: Path,
    field_name: str,
    background_img: Any,
    candidate_top_k_by_stage: dict[str, int],
    fig_width: float,
    dpi: int,
    top1_label_fields: list[str],
    top1_label_bg_alpha: float,
    map_plot_setting: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    field = result["fields"].get(field_name)
    if field is None:
        raise ValueError(f"Field '{field_name}' is unavailable for query_id={query_id}.")
    data = field.detach().cpu().numpy() if hasattr(field, "detach") else np.asarray(field)
    nr = result["grid"]["nr_lin"].detach().cpu().numpy()
    nc = result["grid"]["nc_lin"].detach().cpu().numpy()
    xx, yy = np.meshgrid(nc, nr)
    extent = result["grid"]["raw_extent"]
    raw_bounds = result["grid"]["raw_bounds"]
    width = abs(float(extent[1]) - float(extent[0]))
    height = abs(float(extent[2]) - float(extent[3]))
    fig_h = max(2.0, float(fig_width) * height / max(width, 1e-12))
    fig, ax = plt.subplots(figsize=(float(fig_width), fig_h), dpi=int(dpi))

    if background_img is not None:
        bg = background_img.detach().cpu().numpy() if hasattr(background_img, "detach") else np.asarray(background_img)
        if bg.ndim == 3 and bg.shape[0] in {1, 3, 4}:
            bg = np.moveaxis(bg, 0, -1)
        ax.imshow(bg, extent=extent, origin="upper", alpha=float(map_plot_setting["background_alpha"]))

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.05, 1e-6)
            vmin, vmax = vmin - pad, vmax + pad

    if bool(map_plot_setting["draw_heatmap"]):
        fill_levels = np.linspace(vmin, vmax, int(map_plot_setting["n_fill_levels"]))
        ax.contourf(
            xx,
            yy,
            data,
            levels=fill_levels,
            cmap=str(map_plot_setting["cmap"]),
            alpha=float(map_plot_setting["heatmap_alpha"]),
        )
    if bool(map_plot_setting["draw_contour"]):
        line_levels = np.linspace(vmin, vmax, int(map_plot_setting["n_line_levels"]))
        ax.contour(
            xx,
            yy,
            data,
            levels=line_levels,
            colors=str(map_plot_setting["contour_color"]),
            linewidths=float(map_plot_setting["contour_line_width"]),
            alpha=float(map_plot_setting["contour_alpha"]),
        )

    gt_coord = torch.as_tensor(bundle["coords_gt"][row_idx], dtype=torch.float32).detach().cpu().numpy()

    stage_summary: dict[str, Any] = {}
    top1_points: list[tuple[float, float, str, str]] = []
    arrow_len = 0.055 * max(width, height)
    for stage in ("grid", "mode", "evo"):
        spec = STAGE_SPECS[stage]
        stage_top_k = int(candidate_top_k_by_stage[stage])
        coords, scores = _finite_stage_candidates(bundle, stage, row_idx, stage_top_k)
        stats = _candidate_window_stats(coords, raw_bounds)
        stage_summary[stage] = {
            "coords_key": spec["coords_key"],
            "scores_key": spec["scores_key"],
            "candidate_top_k": stage_top_k,
            "window_stats": stats,
            "top1_coord": coords[0].tolist() if coords.size else None,
            "top1_score": None if scores is None or len(scores) == 0 else float(scores[0]),
        }
        if coords.size == 0:
            continue
        ax.scatter(
            coords[:, 1],
            coords[:, 0],
            marker=spec["marker"],
            s=28,
            c=spec["color"],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.72,
            zorder=10,
            label=f"{spec['label']} candidates",
        )
        top1 = coords[0]
        ax.scatter(
            [float(top1[1])],
            [float(top1[0])],
            marker=spec["marker"],
            s=32,#42
            c=spec["color"],
            edgecolors="white",
            linewidths=0.96,#1.2
            zorder=14,
            label=f"{spec['label']} top1",
        )
        dx, dy = _pose_arrow_delta(float(top1[2]), arrow_len)
        ax.arrow(
            float(top1[1]),
            float(top1[0]),
            dx,
            dy,
            width=0.0015 * max(width, height),
            head_width=0.018 * max(width, height),
            head_length=0.024 * max(width, height),
            length_includes_head=True,
            color=spec["color"],
            ec="black",
            linewidth=0.35,
            zorder=15,
        )
        top1_score = None if scores is None or len(scores) == 0 else float(scores[0])
        top1_label = _format_top1_label(
            fields=top1_label_fields,
            stage_label=spec["label"],
            coord=top1,
            score=top1_score,
        )
        if top1_label is not None:
            ax.text(
                float(top1[1]),
                float(top1[0]),
                f" {top1_label}",
                fontsize=6.5,
                color="white",
                ha="left",
                va="bottom",
                zorder=16,
                bbox={"facecolor": "black", "alpha": top1_label_bg_alpha, "pad": 1.5, "edgecolor": "none"},
            )
        top1_points.append((float(top1[1]), float(top1[0]), spec["color"], spec["label"]))

    if len(top1_points) >= 2:
        ax.plot(
            [p[0] for p in top1_points],
            [p[1] for p in top1_points],
            color="white",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
            zorder=13,
        )
        ax.plot(
            [p[0] for p in top1_points],
            [p[1] for p in top1_points],
            color="black",
            linewidth=0.75,
            linestyle="--",
            alpha=0.9,
            zorder=13,
        )

    gt_dx, gt_dy = _pose_arrow_delta(float(gt_coord[2]), arrow_len)
    ax.arrow(
        float(gt_coord[1]),
        float(gt_coord[0]),
        gt_dx,
        gt_dy,
        width=0.0015 * max(width, height),
        head_width=0.018 * max(width, height),
        head_length=0.024 * max(width, height),
        length_includes_head=True,
        color="#f4e84a",
        ec="#1f1f1f",
        linewidth=0.55,
        zorder=30,
    )
    ax.scatter(
        [float(gt_coord[1])],
        [float(gt_coord[0])],
        marker="*",
        s=170,
        facecolor="#f4e84a",
        edgecolor="#1f1f1f",
        linewidth=1.2,
        zorder=31,
        label="GT",
    )

    title = f"Stage3 search projection | query {query_id}"
    if int(energy_query_id) != int(query_id):
        title += f" | source {int(energy_query_id)}"
    ax.set_title(title, fontsize=9)
    ax.set_xlim(float(extent[0]), float(extent[1]))
    ax.set_ylim(float(extent[2]), float(extent[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#303030")
    ax.legend(
        loc="upper right",
        fontsize=6.5,
        framealpha=0.75,
        facecolor="black",
        labelcolor="white",
        markerscale=0.75,
        handlelength=1.1,
        handletextpad=0.35,
        borderpad=0.35,
    )
    fig.tight_layout(pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "query_id": int(query_id),
        "energy_query_id": int(energy_query_id),
        "row_idx": int(row_idx),
        "output_path": str(output_path),
        "field_name": field_name,
        "n_nr": int(data.shape[0]),
        "n_nc": int(data.shape[1]),
        "raw_bounds": _jsonable(raw_bounds),
        "gt_coord": gt_coord.tolist(),
        "candidate_top_k_by_stage": dict(candidate_top_k_by_stage),
        "top1_label_fields": list(top1_label_fields),
        "top1_label_bg_alpha": float(top1_label_bg_alpha),
        "map_plot_setting": _jsonable(map_plot_setting),
        "stages": stage_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Stage3 grid/mode/evo 4D candidates projected on a 2D INGP energy field.",
    )
    parser.add_argument("--p-yaml", default="/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_visloc.yaml", type=Path, help="Stage3 yaml used to initialize runtime/checkpoints.")
    parser.add_argument("--run-dir",
                        default="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/res_epoch990_R1=88_MidErr=73m/stage3_triplets_test_n149_20260418_055340",
                        type=Path, help="Directory containing stage3_retrieval_bundle.pt.")
    parser.add_argument("--output-dir", type=Path,
                        default="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_visloc03_interval82_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW19_mlpH1024B2_PN1cubie/res_epoch990_R1=88_MidErr=73m/stage3_triplets_test_n149_20260418_055340/searchProcessVis",
                        help="Default: <run-dir>/stage3_search_projection_2d.")
    # parser.add_argument("--p-yaml", default="/home/data/zwk/pyproj_neuloc_v0/trainer_depends/configs/stage3_wingtra.yaml", type=Path, help="Stage3 yaml used to initialize runtime/checkpoints.")
    # parser.add_argument("--run-dir",
    #                     default="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/res_epoch999_wDiffOverlap/stage3_triplets_test_n238_20260511_032557_overlap000",
    #                     type=Path, help="Directory containing stage3_retrieval_bundle.pt.")
    # parser.add_argument("--output-dir", type=Path,
    #                     default="/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_zurich_interval91_wRejectSampling_tripleLoss_singleEdge_hardest_fmMask_dinov2_adF4_codebookW18_mlpH1024B2_PN1cubie/res_epoch999_wDiffOverlap/stage3_triplets_test_n238_20260511_032557_overlap000/searchProcessVis",
    #                     help="Default: <run-dir>/stage3_search_projection_2d.")
    parser.add_argument("--query-ids", default=[5,15,25,35,45,55,65,75,85,95], type=int, nargs="+", help="Bundle row ids / dataset query ids to draw.") #[10,20,30,40,50,60,70,80,90,100]
    parser.add_argument("--candidate-top-k", type=int, default=16, help="Number of candidates per stage to draw.")
    parser.add_argument("--candidate-top-k-grid", type=int, default=512, help="Override top-k for grid candidates.")
    parser.add_argument("--candidate-top-k-mode", type=int, default=32, help="Override top-k for mode candidates.")
    parser.add_argument("--candidate-top-k-evo", type=int, default=5, help="Override top-k for evo candidates.")
    parser.add_argument("--use-train-uav", action="store_true", help="Use train UAV split instead of test split.")
    parser.add_argument("--field-name", default="energy_ingp", choices=("energy_ingp", "energy_visencoder"), help="Field to draw.")
    #绘图控制参数
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=5.8)
    parser.add_argument("--no-sat-background", action="store_true")
    parser.add_argument("--draw-heatmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-contour", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap for the energy heatmap.")
    parser.add_argument("--heatmap-alpha", type=float, default=0.55) #org=0.55
    parser.add_argument("--n-fill-levels", type=int, default=64)
    parser.add_argument("--n-line-levels", type=int, default=24)
    parser.add_argument("--contour-alpha", type=float, default=0.7) #org=0.7
    parser.add_argument("--contour-color", default="#f7f0da")
    parser.add_argument("--contour-line-width", type=float, default=0.55)
    parser.add_argument("--background-alpha", type=float, default=1.0)
    parser.add_argument(
        "--top1-label-fields",
        default="none",
        help="Comma-separated Top1 label fields: stage,row,col,r,s,score,none.",
    )
    parser.add_argument("--top1-label-bg-alpha", type=float, default=0.55, help="Top1 label background alpha in [0, 1].")
    parser.add_argument("--stage3-arg", action="append", default=[], help="Extra raw arg forwarded to Stage3 parser.")
    args =  parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    p_yaml = args.p_yaml.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run dir not found: {run_dir}")
    if not p_yaml.is_file():
        raise FileNotFoundError(f"Stage3 yaml not found: {p_yaml}")
    candidate_top_k_by_stage = _resolve_candidate_top_k_by_stage(args)
    top1_label_fields = _parse_top1_label_fields(args.top1_label_fields)
    map_plot_setting = _build_map_plot_setting(args)
    if not 0.0 <= float(args.top1_label_bg_alpha) <= 1.0:
        raise ValueError("--top1-label-bg-alpha must be in [0, 1].")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "stage3_search_projection_2d"
    bundle = _load_bundle(run_dir)
    n_bundle = int(torch.as_tensor(bundle["coords_gt"]).shape[0])
    source_indices = _load_source_indices(run_dir, n_bundle=n_bundle)
    run_row_meta = _load_run_row_meta(run_dir)
    bad_ids = [query_id for query_id in args.query_ids if query_id < 0 or query_id >= n_bundle]
    if bad_ids:
        raise IndexError(f"query id out of bundle range [0, {n_bundle - 1}]: {bad_ids}")

    inferencer = _build_inferencer(p_yaml=p_yaml, stage3_args=list(args.stage3_arg))
    _init_runtime(inferencer, use_train_uav=bool(args.use_train_uav))
    _normalize_energy_field_local_zoom_option(inferencer)
    analyzer = Stage3EnergyFieldAnalyzer(inferencer)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "bundle_path": str(run_dir / "stage3_retrieval_bundle.pt"),
        "p_yaml": str(p_yaml),
        "output_dir": str(output_dir),
        "use_train_uav": bool(args.use_train_uav),
        "query_ids": [int(query_id) for query_id in args.query_ids],
        "query_id_semantics": "bundle_row_id",
        "energy_query_id_source": "bundle_row_gt_and_exported_query_path",
        "candidate_top_k": int(args.candidate_top_k),
        "candidate_top_k_by_stage": dict(candidate_top_k_by_stage),
        "top1_label_fields": list(top1_label_fields),
        "top1_label_bg_alpha": float(args.top1_label_bg_alpha),
        "map_plot_setting": _jsonable(map_plot_setting),
        "items": [],
        "warnings": [],
    }

    for query_id in args.query_ids:
        energy_query_id, energy_query_meta = _resolve_energy_query_id(
            inferencer=inferencer,
            bundle=bundle,
            row_idx=int(query_id),
            source_indices=source_indices,
            use_train_uav=bool(args.use_train_uav),
        )
        print(
            f"[Stage3SearchProjection] query_id={query_id} energy_query_id={energy_query_id} "
            f"source=bundle.coords_gt query_image="
            f"{'stage3_triplets_summary.csv/query_path' if int(query_id) in run_row_meta else 'dataset_split_row'}"
        )
        result, job = _compute_energy_result_from_bundle_row(
            analyzer=analyzer,
            inferencer=inferencer,
            bundle=bundle,
            row_idx=int(query_id),
            row_meta=run_row_meta.get(int(query_id)),
            use_train_uav=bool(args.use_train_uav),
            render_background=not bool(args.no_sat_background),
        )
        field_name = args.field_name or ("energy_visencoder" if job["ref_feat_backend"] == "visual_encoder" else "energy_ingp")
        zoom_suffix = _format_zoom_suffix_for_filename(job["local_zoom_wh"])
        output_path = output_dir / f"searchProjVis_query{int(query_id):05d}_{zoom_suffix}.png"
        item = _render_projection(
            result=result,
            bundle=bundle,
            row_idx=int(query_id),
            query_id=int(query_id),
            energy_query_id=int(query_id),
            output_path=output_path,
            field_name=field_name,
            background_img=None if args.no_sat_background else job["sat_patch_np"],
            candidate_top_k_by_stage=candidate_top_k_by_stage,
            fig_width=float(args.fig_width),
            dpi=int(args.dpi),
            top1_label_fields=top1_label_fields,
            top1_label_bg_alpha=float(args.top1_label_bg_alpha),
            map_plot_setting=map_plot_setting,
        )
        item["split_index_probe"] = _jsonable(energy_query_meta)
        item["energy_field_source"] = {
            "coord_source": job.get("energy_coord_source"),
            "query_image_source": job.get("query_image_source"),
        }
        for stage, stage_item in item["stages"].items():
            outside = int(stage_item["window_stats"]["outside"])
            if outside > 0:
                stage_top_k = int(stage_item["candidate_top_k"])
                warning = f"query_id={query_id} stage={stage} has {outside} top-{stage_top_k} candidates outside local window"
                print(f"WARNING: {warning}")
                summary["warnings"].append(warning)
        summary["items"].append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage3_search_projection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=True, indent=2)
    print(f"[Stage3SearchProjection] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
