#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Stage-1 train/test unit.

This file is intentionally small.  It keeps only the core path:
1. build frozen/trainable visual encoder + aggregator
2. train with UAV query / satellite positive / satellite negatives
3. test by building a simple satellite gallery and reporting Recall@K

Excluded on purpose: ANCE, gallery-bank cache/artifacts, advanced eval reports,
NetVLAD data-dependent init, code backup, and control-experiment branches.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch
from continuloc.evaluation.partitioned_recall import PartitionedRecall
from continuloc.evaluation.query_rotation_search import (
    absolute_heading_error_rad,
    attach_heading_errors,
    require_scene_mean_scale,
    resolve_retrieval_mode,
    rotation_search_ranking,
    rotation_steps_rad,
    summarize_heading_errors,
)
import torch.nn.functional as F
import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from continuloc.engine.components import NetworkComponents
from continuloc.engine.trainer_base import BaseTrainer
from continuloc.engine.util_stage1_multi_scene_dataloader import MultiSceneDataLoader


class MinimalStage1VisualEncoderTrainer(BaseTrainer):
    """A readable Stage-1 trainer with only the essential train/test logic."""

    def __init__(self, opt=None):
        super().__init__(opt)
        self._init_networks()
        self._setup_trainable_params()

    def _get_train_log_filename(self, exp_name):
        return f"{exp_name}.log"

    def _init_networks(self):
        components = NetworkComponents(self.opt, self.device)
        self.vis_encoder = components.create_visual_encoder()
        self.feat_patch_dim = self.vis_encoder.output_channel
        self.vis_aggregator = components.create_aggregator(self.feat_patch_dim)
        self.feat_q_dim = int(getattr(self.vis_aggregator, "output_dim", self.feat_patch_dim))

    def _setup_trainable_params(self):
        freeze_backbone = bool(getattr(self.opt, "freeze_backbone", True))
        adapter_enabled = bool(getattr(self.opt, "adapter_config", {}).get("enabled", False))
        if freeze_backbone and not adapter_enabled:
            for param in self.vis_encoder.parameters():
                param.requires_grad = False

        self.param2optimize = {"vis_aggregator": self.vis_aggregator}
        self.param2freeze = {}
        if any(param.requires_grad for param in self.vis_encoder.parameters()):
            self.param2optimize["vis_encoder"] = self.vis_encoder
        else:
            self.param2freeze["vis_encoder"] = self.vis_encoder

        print("Stage1 minimal trainable modules:", ", ".join(self.param2optimize.keys()))
        print("Stage1 minimal frozen modules:", ", ".join(self.param2freeze.keys()) or "(none)")

    def _autocast_context(self):
        use_amp = bool(getattr(self.opt, "autocast", False)) and self.device.type == "cuda"
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _encode_for_train(self, imgs):
        if any(param.requires_grad for param in self.vis_encoder.parameters()):
            return self.vis_encoder(imgs)
        with torch.no_grad():
            return self.vis_encoder(imgs)

    def _init_train_dataloader(self):
        from continuloc.datasets.dataset_neuloc_4d_uav_sat_pair import (
            UAVSatPairDataset,
            UAVSatBatchCollator,
        )

        opt = self.opt
        pair_dataloaders = {}
        for scene in opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset_train = self.uav_datasets_train[scene_name]

            n_global_negatives = opt.n_global_negatives_per_batch

            pair_dataset = UAVSatPairDataset(
                uav_dataset=uav_dataset_train,
                sat_dataset=sat_dataset,
                device=self.device,
                sat_as_query=bool(getattr(opt, "sat_as_query", False)),
                pair_alignment_mode=str(getattr(opt, "pair_alignment_mode", "full_4d")),
            )
            pair_dataset.weight = len(uav_dataset_train)

            pair_dataloaders[scene_name] = torch.utils.data.DataLoader(
                pair_dataset,
                batch_size=int(opt.batchsize_uav),
                num_workers=int(opt.num_worker),
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                collate_fn=UAVSatBatchCollator(
                    sat_dataset, n_global_negatives,
                    reject_sampling=bool(getattr(opt, 'reject_sampling', False)),
                ),
                persistent_workers=(int(opt.num_worker) > 0),
            )
            sampler = pair_dataloaders[scene_name].collate_fn.sampler
            if sampler is not None:
                self.logger.info("%s: negative sampling=%s, admissible regions=%d, grid_sha256=%s",
                                 scene_name, "grid" if sampler.grid else "uniform",
                                 len(sampler.rectangles), sampler.source_hash)
            self.logger.info(
                "%s: %d pairs, %d batches, global_negatives_per_batch=%d",
                scene_name,
                len(pair_dataset),
                len(pair_dataloaders[scene_name]),
                n_global_negatives,
            )

        self.dataloader_train = MultiSceneDataLoader(
            pair_dataloaders,
            sampling_strategy=self.opt.scenes_setting.get("sampling_strategy", "round_robin"),
        )

    def _init_loss(self):
        loss_type = str(getattr(self.opt, "loss_type", "triplet_single_edge")).lower()
        if loss_type == "infonce":
            from continuloc.losses.infonce_loss import InfoNCE
            self.loss_type = "infonce"
            self.loss_fn = InfoNCE(
                temperature=float(getattr(self.opt, "infonce_temperature", 0.1)),
                negative_mode=str(getattr(self.opt, "infonce_negative_mode", "batch_and_explicit")),
                learnable_temperature=bool(getattr(self.opt, "infonce_learnable_temperature", False)),
                gate=str(getattr(self.opt, "infonce_gate", "soft")),
            ).to(self.device)
        elif loss_type in {"triplet_single_edge", "tripletloss_singleedge", "triplet_loss",
                           "tripletloss_singleedge_hardest_fm_mask", "tripleloss_singleedge_hardest_fm_mask"}:
            from continuloc.losses.triplet_single_edge import tripletLoss_singleEdge
            self.loss_type = "triplet_single_edge"
            self.loss_fn = tripletLoss_singleEdge(
                alpha=float(getattr(self.opt, "triplet_alpha", 5.0)),
                margin=float(getattr(self.opt, "triplet_margin", 0.0)),
                gate=str(getattr(self.opt, "triplet_gate", "softplus")),
                enable_row_filter=bool(getattr(self.opt, "triplet_enable_row_filter", False)),
                learnable_alpha=bool(getattr(self.opt, "triplet_learnable_alpha", False)),
                metric=str(getattr(self.opt, "triplet_metric", "dist")),
                reduction=str(getattr(self.opt, "triplet_reduction", "mean")),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported loss_type={loss_type!r}; use triplet_single_edge or infonce")
        self.param2optimize.pop("loss_fn", None)
        self.param2freeze.pop("loss_fn", None)
        owner = self.param2optimize if any(p.requires_grad for p in self.loss_fn.parameters()) else self.param2freeze
        owner["loss_fn"] = self.loss_fn

    def _extract_batch(self, batch):
        uavimgs = batch["uavimgs"].to(self.device)
        satimgs_pos = batch["satimgs_pos"].to(self.device)

        if "satimgs_query" in batch:
            uavimgs = torch.cat([uavimgs, batch["satimgs_query"].to(self.device)], dim=0)
            satimgs_pos = torch.cat([satimgs_pos, batch["satimgs_pos2satimg_query"].to(self.device)], dim=0)

        satimgs_neg = batch.get("satimgs_neg", None)
        if satimgs_neg is not None:
            satimgs_neg = satimgs_neg.to(self.device)
            if satimgs_neg.ndim != 4:
                raise ValueError("Global negatives must have shape [N, C, H, W]")

        return uavimgs, satimgs_pos, satimgs_neg

    def _forward_train_batch(self, uavimgs, satimgs_pos, satimgs_neg):
        refs = [satimgs_pos]
        if satimgs_neg is not None:
            refs.append(satimgs_neg)
        imgs_input = torch.cat([uavimgs] + refs, dim=0)

        feats_patch = self._encode_for_train(imgs_input)
        feats = self.vis_aggregator(feats_patch)

        batch_size = uavimgs.shape[0]
        feats_q = feats[:batch_size]
        feats_ref = feats[batch_size:]
        feat_dist_mat = torch.norm(feats_q[:, None, :] - feats_ref[None, :, :], p=2, dim=-1)
        return feats_q, feats_ref, feat_dist_mat

    @staticmethod
    def _build_pos_mask(batch_size, num_refs, device):
        pos_mask = torch.zeros(batch_size, num_refs, dtype=torch.bool, device=device)
        pos_mask[:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=device)
        return pos_mask

    def _compute_loss(self, feats_q, feats_ref, feat_dist_mat):
        batch_size = feats_q.shape[0]
        feats_pos = feats_ref[:batch_size]
        feats_neg = feats_ref[batch_size:]

        if self.loss_type == "infonce":
            return self.loss_fn(
                feats_q, feats_pos, explicit_negative_keys=feats_neg,
                metric=str(getattr(self.opt, "infonce_metric", "similarity")),
                normalize=bool(getattr(self.opt, "infonce_normalize", True)),
            )

        pos_mask = self._build_pos_mask(batch_size, feats_ref.shape[0], feat_dist_mat.device)
        feat_mat = feats_q @ feats_ref.transpose(0, 1) if self.loss_fn.metric == "similarity" else feat_dist_mat
        return self.loss_fn(feat_mat, pos_mask)

    def _checkpoint_modules(self):
        return {**self.param2optimize, **self.param2freeze}

    def train(self):
        opt = self.opt
        use_amp = bool(getattr(opt, "autocast", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        self._init_loss()
        from continuloc.engine.util_mk_optimizer import create_optimizer_w_temple

        self.optimizer = create_optimizer_w_temple(self.param2optimize, "adam", opt=opt)
        begin_epoch = self._load_checkpoint(
            getattr(opt, "load2train", ""),
            self._checkpoint_modules(),
            self.optimizer,
            mode="train",
        )

        self._init_logger()
        self._init_datasets(create_train_loader=False)
        self._init_train_dataloader()

        step = 0
        since = time.time()
        for epoch in range(begin_epoch, int(opt.num_epochs)):
            self.logger.info("Epoch %d/%d", epoch, int(opt.num_epochs) - 1)
            last_loss = None

            for it, batch in tqdm.tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train)):
                uavimgs, satimgs_pos, satimgs_neg = self._extract_batch(batch)
                with self._autocast_context():
                    feats_q, feats_ref, feat_dist_mat = self._forward_train_batch(
                        uavimgs,
                        satimgs_pos,
                        satimgs_neg,
                    )
                    loss = self._compute_loss(feats_q, feats_ref, feat_dist_mat)

                self.optimizer.zero_grad()
                if use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if it % 10 == 0:
                    batch_size = feats_q.shape[0]
                    recall1 = (
                        torch.argmin(feat_dist_mat, dim=-1)
                        == torch.arange(batch_size, device=feat_dist_mat.device)
                    ).float().mean()
                    self.logger.info(
                        "iter=%d scene=%s loss=%.6f train_recall@1=%.4f",
                        it,
                        batch.get("scene_name", "unknown"),
                        float(loss.detach().item()),
                        float(recall1.detach().item()),
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("loss_it", loss.item(), step)
                        self.writer.add_scalar("train_recall1_it", recall1.item(), step)

                last_loss = loss
                step += 1

            save_freq = max(1, int(getattr(opt, "save_freq", 1)))
            if ((epoch + 1) % save_freq == 0) or (epoch == int(opt.num_epochs) - 1):
                self._save_checkpoint(epoch, self._checkpoint_modules(), self.optimizer)

            if bool(getattr(opt, "val", False)):
                val_freq = max(1, int(getattr(opt, "val_freq", 1)))
                if ((epoch + 1) % val_freq == 0) or (epoch == int(opt.num_epochs) - 1):
                    self.test(init_datasets=False, load_ckpt=False, restore_train=True)

            elapsed = time.time() - since
            since = time.time()
            if last_loss is not None:
                self.logger.info("epoch=%d loss=%.6f elapsed=%.0fm %.0fs", epoch, last_loss.item(), elapsed // 60, elapsed % 60)

        self.logger.info("Stage1 minimal training finished.")

    def _load_eval_ckpt_if_needed(self):
        ckpt_path = getattr(self.opt, "load2test", "") or getattr(self.opt, "load2train", "")
        if ckpt_path:
            self._load_checkpoint(ckpt_path, self._checkpoint_modules(), optimizer=None, mode="test")
            print(f"Loaded eval checkpoint: {ckpt_path}")

    def _encode_imgs_eval(self, imgs):
        feats = self._get_feats_fm_imgs(imgs)
        return F.normalize(feats, dim=-1)

    @staticmethod
    def _warp_uav_imgs(imgs, rot_rad=None, scale_f=None):
        if rot_rad is None and scale_f is None:
            return imgs
        bsz = imgs.shape[0]
        device = imgs.device
        dtype = imgs.dtype
        if rot_rad is None:
            rot_rad = torch.zeros(bsz, device=device, dtype=dtype)
        else:
            rot_rad = rot_rad.to(device=device, dtype=dtype)
        if scale_f is None:
            scale_f = torch.ones(bsz, device=device, dtype=dtype)
        else:
            scale_f = scale_f.to(device=device, dtype=dtype)

        theta = torch.zeros(bsz, 2, 3, device=device, dtype=dtype)
        cos_v = torch.cos(rot_rad)
        sin_v = torch.sin(rot_rad)
        theta[:, 0, 0] = cos_v * scale_f
        theta[:, 0, 1] = sin_v * scale_f
        theta[:, 1, 0] = -sin_v * scale_f
        theta[:, 1, 1] = cos_v * scale_f
        grid = F.affine_grid(theta, imgs.size(), align_corners=False)
        return F.grid_sample(imgs, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _build_scene_gallery(self, sat_dataset, overlap, chunk_size):
        gallery_scale = float(sat_dataset.satimgsize_scale_to_ref_m_mean)
        mode = str(getattr(self.opt, "val_gallery_scale_mode", "scene_mean"))
        if mode == "scene_mean":
            scales = [gallery_scale]
            crop_sizes = [float(sat_dataset.satimgsize2crop_mean)]
        elif mode == "multiscale":
            levels = int(getattr(self.opt, "val_gallery_scale_levels", 3))
            if levels < 2:
                raise ValueError("multiscale requires at least two scale levels")
            scales, crop_sizes = sat_dataset.mk_sacle_levels(
                n_level=levels,
                scale_mode=getattr(self.opt, "val_gallery_scale_spacing", "linear"),
            )
        else:
            raise ValueError(f"Unknown val_gallery_scale_mode: {mode}")
        if chunk_size < 1 or not 0 <= overlap < 1:
            raise ValueError("chunk_size must be positive and overlap must be in [0, 1)")
        coords = []
        for scale, crop_size in zip(scales, crop_sizes):
            nrcs = sat_dataset.crop_sat_unifrom(
                size2clip=float(crop_size), overlap=float(overlap), only_nrcs=True,
            )
            nrcs = torch.as_tensor(nrcs, dtype=torch.float32).flatten(0, 1)
            coords.append(torch.cat([
                nrcs, torch.zeros(len(nrcs), 1),
                torch.full((len(nrcs), 1), float(scale)),
            ], dim=-1))
        coords_gallery = torch.cat(coords)
        if not len(coords_gallery):
            raise ValueError("Empty satellite gallery; check scene extent and crop scales")
        print(f"Gallery: mode={mode}, scales={[float(s) for s in scales]}, N={len(coords_gallery)}", flush=True)

        feats = []
        with torch.no_grad():
            for start in range(0, coords_gallery.shape[0], int(chunk_size)):
                end = min(start + int(chunk_size), coords_gallery.shape[0])
                coords_chunk = coords_gallery[start:end]
                if hasattr(sat_dataset, "crop_satimg_by_4d_coords_fast"):
                    satimgs = sat_dataset.crop_satimg_by_4d_coords_fast(
                        coords_chunk,
                        apply_rotation=False,
                        chunk_size=int(chunk_size),
                    )
                else:
                    satimgs = sat_dataset.crop_satimg_by_4d_coords(coords_chunk, apply_rotation=False)
                feats.append(self._encode_imgs_eval(satimgs.to(self.device)).cpu())
        return coords_gallery.cpu(), torch.cat(feats, dim=0), gallery_scale

    def _rotation_search_responses(self, uavimgs, scale_f, warp_angles, feats_gallery, chunk_size):
        """Cosine responses for each query rotation. Both feature sets are L2-normalized."""
        batch_size = uavimgs.shape[0]
        responses = torch.empty(
            batch_size, warp_angles.numel(), feats_gallery.shape[0], dtype=torch.float32,
        )
        gallery = feats_gallery.to(device=uavimgs.device)
        angles = warp_angles.to(device=uavimgs.device, dtype=uavimgs.dtype)
        with torch.no_grad():
            for angle_index, angle in enumerate(angles):
                warped = self._warp_uav_imgs(
                    uavimgs, rot_rad=angle.expand(batch_size), scale_f=scale_f,
                )
                encoded = []
                for start in range(0, batch_size, int(chunk_size)):
                    encoded.append(self._encode_imgs_eval(warped[start:start + int(chunk_size)]))
                feats_q = torch.cat(encoded, dim=0)
                gallery_match = gallery.to(dtype=feats_q.dtype)
                responses[:, angle_index] = (feats_q @ gallery_match.transpose(0, 1)).detach().cpu()
        return responses

    def test(self, init_datasets=True, load_ckpt=True, restore_train=True):
        """Recall@K against a north-aligned satellite gallery.

        ``north_aligned`` rotates each query with its recorded heading.
        ``query_rotation_search`` scales queries to the scene mean and sums
        cosine responses over discrete query rotations.
        """
        if init_datasets or not hasattr(self, "sat_datasets"):
            self._init_datasets(create_train_loader=False)
        if load_ckpt:
            self._load_eval_ckpt_if_needed()

        models = list(self._checkpoint_modules().values())
        old_modes = [model.training for model in models]
        for model in models:
            model.eval()

        overlap = float(getattr(self.opt, "val_overlap", 0.5))
        chunk_size = int(getattr(self.opt, "val_chunk_size", 1024))
        k_values = [1, 5, 10, 20, 50]
        retrieval_mode = resolve_retrieval_mode(getattr(self.opt, "retrieval_mode", "north_aligned"))
        query_rot2uniform = bool(getattr(self.opt, "val_query_rot2uniform", True))
        query_scale2uniform = bool(getattr(self.opt, "val_query_scale2uniform", False))
        warp_angles = None
        if retrieval_mode == "query_rotation_search":
            require_scene_mean_scale(getattr(self.opt, "val_gallery_scale_mode", "scene_mean"))
            step_deg = getattr(self.opt, "query_rotation_step_deg", 10.0)
            warp_angles = rotation_steps_rad(step_deg)
            print(
                f"Retrieval: query_rotation_search, step_deg={float(step_deg):g}, angles={warp_angles.numel()}",
                flush=True,
            )
        results = {}

        for scene in self.opt.scenes_setting["scenes"]:
            scene_name = scene["name"]
            sat_dataset = self.sat_datasets[scene_name]
            uav_dataset = self.uav_datasets_eval[scene_name]
            coords_gallery, feats_gallery, gallery_scale = self._build_scene_gallery(
                sat_dataset,
                overlap=overlap,
                chunk_size=chunk_size,
            )

            uav_loader = torch.utils.data.DataLoader(
                uav_dataset,
                batch_size=int(self.opt.batchsize_uav),
                num_workers=int(getattr(self.opt, "num_worker_eval", 0)),
                shuffle=False,
                drop_last=False,
                pin_memory=False,
            )

            metrics = PartitionedRecall(uav_dataset.eval_split_labels, k_values)
            top_k = min(max(k_values), feats_gallery.shape[0])
            heading_errors = []
            for batch in tqdm.tqdm(uav_loader, desc=f"test {scene_name}"):
                uavimgs, coords_uav = batch[0], batch[1]
                uavimgs = uavimgs.to(self.device)
                coords_uav = coords_uav.to(self.device)

                if retrieval_mode == "query_rotation_search":
                    scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6)
                    responses = self._rotation_search_responses(
                        uavimgs, scale_f, warp_angles, feats_gallery, chunk_size,
                    )
                    indices, heading_hat = rotation_search_ranking(responses, warp_angles, top_k)
                    heading_errors.append(absolute_heading_error_rad(heading_hat, coords_uav[:, 2].cpu()))
                else:
                    rot_align = -coords_uav[:, 2] if query_rot2uniform else None
                    scale_f = gallery_scale / coords_uav[:, 3].clamp(min=1e-6) if query_scale2uniform else None
                    if query_rot2uniform or query_scale2uniform:
                        uavimgs = self._warp_uav_imgs(uavimgs, rot_rad=rot_align, scale_f=scale_f)

                    with torch.no_grad():
                        feats_q = self._encode_imgs_eval(uavimgs).cpu()

                    dist_feat = torch.cdist(feats_q, feats_gallery)
                    indices = torch.topk(dist_feat, k=top_k, dim=1, largest=False).indices
                coords_topk = coords_gallery[indices]
                dist_nrc = torch.norm(coords_uav[:, None, :2].cpu() - coords_topk[:, :, :2], p=2, dim=-1)
                hits = dist_nrc < float(sat_dataset.halfimg_radius_nrc)

                metrics.update(hits, dist_nrc[:, 0])

            nrc2meter = float(sat_dataset.halfimg_radius_meter) / max(float(sat_dataset.halfimg_radius_nrc), 1e-8)
            scene_result = metrics.finish(nrc2meter, scene_name, self.logger)
            if heading_errors:
                summary = summarize_heading_errors(
                    uav_dataset.eval_split_labels, torch.cat(heading_errors),
                )
                scene_result = attach_heading_errors(scene_result, summary)
                for split_name, stats in summary.items():
                    text = (
                        f"[{scene_name}][{split_name}] heading_error "
                        f"mean={stats['heading_error_deg_mean']:.3f}° "
                        f"median={stats['heading_error_deg_median']:.3f}°"
                    )
                    print(text)
                    if self.logger is not None:
                        self.logger.info(text)
            results[scene_name] = scene_result

        if restore_train:
            for model, was_train in zip(models, old_modes):
                model.train(was_train)
        return results


VisualEncoderTrainer = MinimalStage1VisualEncoderTrainer


