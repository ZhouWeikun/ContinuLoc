"""Fixed-rotation/scale Stage-3 spatial response slices.

Extracted from master cfb8d0724bf68bcab2c888aff790bc9c126711a9,
trainers/util_stage3_energy_field_analyzer.py: build_grid and compute_fields.
Preserves exp(-L2(normalized query, normalized reference)); removes the obsolete
projector/FFT/rendering branches and forwards scoring chunk size explicitly.
Runtime/model loading uses the existing MinimalStage3INGPInferencer.
"""
import numpy as np
import torch
import torch.nn.functional as TF


class Stage3EnergyFieldAnalyzer:
    """Sample a 2D location grid while fixing rotation and scale to coord_q."""

    def __init__(self, trainer):
        self.trainer = trainer

    @staticmethod
    def normalize_ref_feat_backend(ref_feat_backend=None, use_vis_ref=None):
        if ref_feat_backend is None:
            if use_vis_ref is not None:
                ref_feat_backend = "visual_encoder" if bool(use_vis_ref) else "ingp"
            else:
                ref_feat_backend = "ingp"

        ref_feat_backend = str(ref_feat_backend).strip().lower()
        if ref_feat_backend not in {"ingp", "visual_encoder"}:
            raise ValueError(
                f"ref_feat_backend must be 'ingp' or 'visual_encoder', got {ref_feat_backend!r}"
            )
        return ref_feat_backend

    def build_grid(
        self,
        coord_q,
        n_nr=128,
        n_nc=128,
        local_zoom_wh=None,
        raw_bounds=None,
        verbose=False,
    ):
        if int(n_nr) < 2 or int(n_nc) < 2:
            raise ValueError("Energy grid dimensions must be >= 2.")
        if local_zoom_wh is not None and (len(local_zoom_wh) != 2 or
                any(not 0 < float(v) <= 1 for v in local_zoom_wh)):
            raise ValueError("local_zoom_wh must be null or two ratios in (0, 1].")
        if local_zoom_wh is not None and raw_bounds is not None:
            raise ValueError("local_zoom_wh and raw_bounds are mutually exclusive.")
        trainer = self.trainer
        global_nr_min = float(trainer.sat_dataset.nr2sample_min)
        global_nr_max = float(trainer.sat_dataset.nr2sample_max)
        global_nc_min = float(trainer.sat_dataset.nc2sample_min)
        global_nc_max = float(trainer.sat_dataset.nc2sample_max)

        if raw_bounds is not None:
            required = {"start_nr", "end_nr", "start_nc", "end_nc"}
            if set(raw_bounds) != required:
                raise ValueError(f"raw_bounds must contain exactly {sorted(required)}.")
            start_nr = float(raw_bounds["start_nr"])
            end_nr = float(raw_bounds["end_nr"])
            start_nc = float(raw_bounds["start_nc"])
            end_nc = float(raw_bounds["end_nc"])
            if not (
                global_nr_min <= start_nr < end_nr <= global_nr_max
                and global_nc_min <= start_nc < end_nc <= global_nc_max
            ):
                raise ValueError(
                    "raw_bounds must be ordered and contained within the scene bounds: "
                    f"nr=[{global_nr_min}, {global_nr_max}], "
                    f"nc=[{global_nc_min}, {global_nc_max}]."
                )
        elif local_zoom_wh is not None:
            w_ratio_nr, w_ratio_nc = local_zoom_wh
            span_nr = global_nr_max - global_nr_min
            span_nc = global_nc_max - global_nc_min
            half_nr = (span_nr * float(w_ratio_nr)) / 2
            half_nc = (span_nc * float(w_ratio_nc)) / 2

            center_nr = float(coord_q[0].item())
            center_nc = float(coord_q[1].item())
            start_nr = max(global_nr_min, center_nr - half_nr)
            end_nr = min(global_nr_max, center_nr + half_nr)
            start_nc = max(global_nc_min, center_nc - half_nc)
            end_nc = min(global_nc_max, center_nc + half_nc)

            if verbose:
                print(f"[Analyze] Local Zoom Enabled: Center=({center_nr:.2f}, {center_nc:.2f})")
                print(f"          Range NR: [{start_nr:.2f}, {end_nr:.2f}], NC: [{start_nc:.2f}, {end_nc:.2f}]")
        else:
            start_nr, end_nr = global_nr_min, global_nr_max
            start_nc, end_nc = global_nc_min, global_nc_max

        if not start_nr < end_nr or not start_nc < end_nc:
            raise ValueError("Energy grid has empty bounds; check query coordinates and scene.")
        nr_lin = torch.linspace(start_nr, end_nr, int(n_nr), device=trainer.device)
        nc_lin = torch.linspace(start_nc, end_nc, int(n_nc), device=trainer.device)
        nr_grid, nc_grid = torch.meshgrid(nr_lin, nc_lin, indexing="ij")
        rot_grid = torch.full_like(nr_grid, coord_q[2])
        scale_grid = torch.full_like(nr_grid, coord_q[3])
        coords_grid = torch.stack([nr_grid, nc_grid, rot_grid, scale_grid], dim=-1)

        span_nr = max(end_nr - start_nr, 1e-12)
        span_nc = max(end_nc - start_nc, 1e-12)
        gt_nr_f = (float(coord_q[0].item()) - start_nr) / span_nr * (int(n_nr) - 1)
        gt_nc_f = (float(coord_q[1].item()) - start_nc) / span_nc * (int(n_nc) - 1)
        gt_index = (
            int(np.clip(round(gt_nr_f), 0, int(n_nr) - 1)),
            int(np.clip(round(gt_nc_f), 0, int(n_nc) - 1)),
        )

        return {
            "nr_lin": nr_lin,
            "nc_lin": nc_lin,
            "coords_grid": coords_grid,
            "coords_flat": coords_grid.view(-1, 4),
            "raw_extent": (start_nc, end_nc, end_nr, start_nr),
            "raw_bounds": {
                "start_nr": start_nr,
                "end_nr": end_nr,
                "start_nc": start_nc,
                "end_nc": end_nc,
            },
            "gt_index": gt_index,
            "gt_raw": (float(coord_q[0].item()), float(coord_q[1].item())),
            "local_zoom_label": (
                "raw_bounds" if raw_bounds is not None
                else "global" if local_zoom_wh is None
                else local_zoom_wh
            ),
        }

    def compute_fields(self, query, grid, plot_mode="ingp", ref_feat_backend="ingp", chunk_size_vis=1024):
        trainer = self.trainer
        plot_mode = str(plot_mode or "ingp").strip().lower()
        if plot_mode != "ingp":
            raise ValueError(f"Only the INGP/visual encoder slice is supported; plot_mode must be 'ingp', got {plot_mode}")
        ref_feat_backend = self.normalize_ref_feat_backend(ref_feat_backend=ref_feat_backend)

        if int(chunk_size_vis) < 1:
            raise ValueError("chunk_size_vis must be positive.")
        feat_vis = query["feat_vis"]
        coords_flat = grid["coords_flat"]
        coords_grid = grid["coords_grid"]

        energy_visencoder = None
        energy_ingp = None
        energy_projector = None

        with torch.no_grad():
            if ref_feat_backend == "visual_encoder":
                feat_q_vis = TF.normalize(feat_vis, dim=-1)
                dist_chunks = []
                for start in range(0, coords_flat.shape[0], int(chunk_size_vis)):
                    end = min(start + int(chunk_size_vis), coords_flat.shape[0])
                    satimgs_refs_chunk = trainer.sat_dataset.crop_satimg_by_4d_coords(coords_flat[start:end].cpu()).to(trainer.device)
                    feats_ref_chunk = TF.normalize(trainer._get_feats_fm_imgs(satimgs_refs_chunk), dim=-1)
                    dist_chunk = torch.norm(feats_ref_chunk - feat_q_vis, dim=-1)
                    dist_chunks.append(dist_chunk)
                dist_visencoder = torch.cat(dist_chunks, dim=0).reshape(*coords_grid.shape[:2])
                energy_visencoder = torch.exp(-dist_visencoder)
            else:
                dist_ingp = trainer._compute_metric_from_query_and_points(
                    metric="dist",
                    feat_type="ingp",
                    query_feats=feat_vis,
                    ref_points=coords_flat,
                    chunk_size=int(chunk_size_vis),
                ).reshape(*coords_grid.shape[:2])
                energy_ingp = torch.exp(-dist_ingp)

        return {
            "energy_visencoder": energy_visencoder,
            "energy_ingp": energy_ingp,
            "energy_projector": energy_projector,
        }
