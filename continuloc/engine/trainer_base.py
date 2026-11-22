#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trainer base class.

Shared functionality includes device management, dataset initialization,
checkpoint management, and logging."""

import torch
import copy
import os
from torch.utils.tensorboard import SummaryWriter

from continuloc.config.parser import DEFAULT_SPLIT_SOURCE, get_parse
from continuloc.engine.logger_utils import get_logger
from continuloc.datasets.uav_satellite import UAVDataset, SatDataset


class BaseTrainer:
    """Base class for all trainers.

    Provides configuration management, device initialization, dataset loading,
    checkpoint loading/saving, and logging."""

    def __init__(self, opt=None):
        """Initialize the base trainer.

        Args:
            opt: Configuration object; call get_parse() when None."""
        # 1. Configuration management
        self.opt = opt if opt is not None else get_parse()

        # 2. Device setup
        self._init_device()

        # 3. Logging is initialized lazily in train().
        self.logger = None
        self.writer = None
        self.exp_dir2save = None
        self.log_dir2save = None
        self.ckpt_dir2save = None
        self._exp_backup_saved = False


    def _init_device(self):
        """Initialize the GPU or CPU device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False

        self.device = device
        print(f"✅ 设备初始化: {self.device}")


    def _get_train_log_filename(self, exp_name):
        """Return the training log filename; subclasses may override this."""
        return "train.log"


    def _init_logger(self, exp_name=None):
        """Initialize logging.

        Args:
            exp_name: Experiment name; use opt.exp_name when None."""
        opt = self.opt

        if exp_name is None:
            exp_name = opt.exp_name

        log_root = getattr(opt, 'dir2save_log', None) or getattr(opt, 'exps_dir', 'gen_fm_exps/logs')
        ckpt_root = getattr(opt, 'dir2save_ckpt', None) or log_root
        os.makedirs(log_root, exist_ok=True)

        exp_name_unique = exp_name
        counter = 0
        while (
            os.path.exists(os.path.join(log_root, exp_name_unique))
            or os.path.exists(os.path.join(ckpt_root, exp_name_unique))
        ):
            counter += 1
            exp_name_unique = f"{exp_name}_{counter}"

        opt.exp_name = exp_name_unique
        self.log_dir2save = os.path.join(log_root, exp_name_unique)
        self.ckpt_dir2save = os.path.join(ckpt_root, exp_name_unique)
        self.exp_dir2save = self.ckpt_dir2save
        os.makedirs(self.log_dir2save, exist_ok=True)

        # Initialize the logger.
        log_filename = self._get_train_log_filename(exp_name_unique)
        log_path = os.path.join(self.log_dir2save, log_filename)
        self.logger = get_logger(log_path, 'trainer_logger')
        self.logger.info(f"日志目录: {self.log_dir2save}")
        self.logger.info(f"日志文件: {log_path}")
        self.logger.info(f"Checkpoint目录: {self.ckpt_dir2save}")

        # Initialize TensorBoard.
        if opt.tensorboard:
            tb_path = os.path.join(self.log_dir2save, "train_tensorboard.log")
            self.writer = SummaryWriter(tb_path)
            self.logger.info(f"Tensorboard日志: {tb_path}")

        print(f"✅ 日志初始化完成: {exp_name_unique}")


    def _init_datasets(self, create_train_loader=False):
        """Initialize datasets for multiple scenes.

        Args:
            create_train_loader: Whether to create a training DataLoader."""
        opt = self.opt
        scenes = opt.scenes_setting['scenes']

        # Choose where satellite tensors are cached/cropped.
        sat_device = self.device
        if getattr(opt, "satmaps_on_cpu", False):
            sat_device = torch.device("cpu")
            msg = "satmaps_on_cpu=True; using CPU for SatDataset cache and cropping."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        elif getattr(opt, "num_worker", 0) > 0 and self.device.type == "cuda":
            sat_device = torch.device("cpu")
            msg = ("num_worker>0 with CUDA SatDataset triggers CUDA init in forked workers; "
                   "using CPU for SatDataset. Set num_worker=0 to keep CUDA dataset ops.")
            if self.logger:
                self.logger.warning(msg)
            else:
                print(f"Warning: {msg}")

        # Store datasets for each scene.
        self.sat_datasets = {}
        self.uav_datasets_train = {}
        self.uav_datasets_test = {}
        self.uav_datasets_val = {}
        self.uav_datasets_eval = {}

        if create_train_loader:
            self.pair_dataloaders = {}

        pad_mode = getattr(opt, "pad_mode", None)

        for scene in scenes:
            scene_name = scene['name']

            # Log scene information.
            log_msg = f"正在初始化场景: {scene_name}"
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)

            # Create SatDataset.
            sat_dataset = SatDataset(
                p_satinfo_json=scene['p_satinfo_json'],
                satellite_filepaths=scene.get('satellite_filepaths'),
                global_negative_sampling_grid=scene.get('global_negative_sampling_grid'),
                p_uav_geocsv=scene['p_uav_geocsv'],
                imgsize2net=opt.imgsize2net,
                split_train_ratio=opt.split_train_ratio,
                split_mode=opt.split_mode,
                split_source=getattr(opt, "split_source", DEFAULT_SPLIT_SOURCE),
                p_split_json=scene.get('p_split_json'),
                enable_edge_filter=getattr(opt, 'enable_edge_filter', True),
                geo_rc_epsg_code=scene.get('geo_rc_epsg_code'),
                name=scene_name,
                device=sat_device,
                enable_scale_filter=getattr(opt, "enable_scale_filter", False),
                scale_filter_mode=getattr(opt, "scale_filter_mode", None),
                sat_scale_sampling_mode=getattr(opt, "sat_scale_sampling_mode", "empirical"),
                sat_scale_jitter_enabled=getattr(opt, "sat_scale_jitter_enabled", True),
                sat_scale_jitter_half_span_ratio=getattr(opt, "sat_scale_jitter_half_span_ratio", 0.05),
                pad_mode=pad_mode,
            )
            self.sat_datasets[scene_name] = sat_dataset

            # Create the training UAVDataset.
            uav_dataset_train = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                uavimgs_dir=scene.get('uavimgs_dir'),
                p_uav_geocsv=scene['p_uav_geocsv'],
                sat_dataset=sat_dataset,
                stage='train',
                query_geometry_method=getattr(opt, 'query_geometry_method', 'pipelined'),
                query_prefilter_resolution_factor=getattr(opt, 'query_prefilter_resolution_factor', 2.0),
                name=scene_name,
                device='cpu',
                pad_mode=pad_mode,
                dataset_name=self.opt.scenes_setting['dataset_name'],
                uav_image_layout=scene.get('uav_image_layout'),
                split_train_ratio=opt.split_train_ratio,
                split_val_ratio=getattr(opt, "split_val_ratio", 0.0),
                split_mode=opt.split_mode,
                split_source=getattr(opt, "split_source", DEFAULT_SPLIT_SOURCE),
                p_split_json=scene.get('p_split_json'),
                enable_edge_filter=getattr(opt, 'enable_edge_filter', True),
            )
            self.uav_datasets_train[scene_name] = uav_dataset_train

            # Create the test UAVDataset.
            uav_dataset_test = UAVDataset(
                p_uavinfo_json=scene['p_uavinfo_json'],
                uavimgs_dir=scene.get('uavimgs_dir'),
                p_uav_geocsv=scene['p_uav_geocsv'],
                sat_dataset=sat_dataset,
                stage='eval',
                name=scene_name,
                pad_mode=pad_mode,
                dataset_name=self.opt.scenes_setting['dataset_name'],
                uav_image_layout=scene.get('uav_image_layout'),
                split_train_ratio=opt.split_train_ratio,
                split_val_ratio=getattr(opt, "split_val_ratio", 0.0),
                split_mode=opt.split_mode,
                split_source=getattr(opt, "split_source", DEFAULT_SPLIT_SOURCE),
                p_split_json=scene.get('p_split_json'),
                enable_edge_filter=getattr(opt, 'enable_edge_filter', True),
            )
            self.uav_datasets_eval[scene_name] = uav_dataset_test
            for split in ("val", "test"):
                view = copy.copy(uav_dataset_test)
                view.switch_stage(split)
                getattr(self, "uav_datasets_" + split)[scene_name] = view
            msg = (f"[{scene_name}] train={uav_dataset_test.n_train}, "
                   f"val={uav_dataset_test.n_val}, test={uav_dataset_test.n_test}; "
                   f"validation source={uav_dataset_test.validation_split_source}")
            print(msg)
            if self.logger:
                self.logger.info(msg)

            # Create the training DataLoader if requested.
            if create_train_loader:
                # Add specialized DataLoader construction here if needed.
                # For example, use UAVSatPairDataset with MultiSceneDataLoader.
                pass

        # Use the first scene as the primary dataset for UDF and related computations.
        first_scene_name = scenes[0]['name']
        self.sat_dataset = self.sat_datasets[first_scene_name]
        self.uav_dataset_train = self.uav_datasets_train[first_scene_name]
        self.uav_dataset_test = self.uav_datasets_test[first_scene_name]
        self.uav_dataset_val = self.uav_datasets_val[first_scene_name]
        self.uav_dataset_eval = self.uav_datasets_eval[first_scene_name]

        if self.logger:
            self.logger.info(f"✅ 数据集初始化完成，共{len(scenes)}个场景")
        else:
            print(f"✅ 数据集初始化完成，共{len(scenes)}个场景")


    def _load_checkpoint(self, ckpt_config, modules_dict, optimizer=None, mode='train'):
        """Load a checkpoint.

        Args:
            ckpt_config: Checkpoint configuration as a dict or string.
            modules_dict: Modules to load, e.g. {'vis_encoder': self.vis_encoder, 'grid': self.grid}.
            optimizer: Required only when resuming training with mode='train'.
            mode: 'train' or 'test'.

        Returns:
            Starting epoch, or 0 for a new training run."""
        from continuloc.engine.checkpoints import load_param

        begin_epoch = 0

        # Handle dictionary checkpoint configuration.
        if isinstance(ckpt_config, dict):
            for module_name, ckpt_path in ckpt_config.items():
                if ckpt_path and ckpt_path != "":
                    if module_name in modules_dict:
                        load_param(ckpt_path, {module_name: modules_dict[module_name]})
                        msg = f"✅ 加载{module_name}模块: {ckpt_path}"
                        if self.logger:
                            self.logger.info(msg)
                        else:
                            print(msg)

        # Handle string checkpoint configuration.
        elif isinstance(ckpt_config, str) and ckpt_config != "":
            # Load all modules.
            params_to_load = {**modules_dict}
            if optimizer is not None and mode == 'train':
                params_to_load['optimizer_state'] = optimizer
            loaded_params = load_param(ckpt_config, params_to_load)

            # Retrieve the epoch information.
            if 'epoch' in loaded_params:
                begin_epoch = loaded_params['epoch'] + 1

            msg = f"✅ 加载checkpoint: {ckpt_config}, 从epoch {begin_epoch}继续训练"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        return begin_epoch


    def _save_checkpoint(self, epoch, modules_dict, optimizer=None):
        """Save a checkpoint.

        Args:
            epoch: Current epoch.
            modules_dict: Modules to save.
            optimizer: Optional optimizer."""
        from continuloc.engine.checkpoints import save_param

        # Prepare the parameters to save.
        params_to_save = {**modules_dict}

        # Add optimizer state and epoch information.
        if optimizer is not None:
            params_to_save['optimizer_state'] = optimizer
        params_to_save['epoch'] = epoch

        # Save the checkpoint.
        save_dir = self.ckpt_dir2save or self.exp_dir2save or self.opt.exp_name
        save_param(save_dir, params_to_save)

        if (not self._exp_backup_saved) and save_dir:
            try:
                from continuloc.engine.git_snapshot import backup_experiment
                backup_experiment(save_dir, self.opt)
                self._exp_backup_saved = True
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"首次checkpoint后备份实验信息失败: {exc}")
                else:
                    print(f"Warning: 首次checkpoint后备份实验信息失败: {exc}")

        if self.logger:
            self.logger.info(f"✅ 已保存checkpoint: epoch {epoch}, dir={save_dir}")


    def _get_feats_fm_imgs(self, imgs):
        """Extract image features.

        Args:
            imgs: Image tensor [B, C, H, W].

        Returns:
            Feature vectors [B, feat_dim]."""
        with torch.no_grad():
            feats_patch = self.vis_encoder(imgs)
            feats = self.vis_aggregator(feats_patch)
        return feats


    def train(self):
        """Run the main training loop.

        Subclasses must implement this method."""
        raise NotImplementedError("子类必须实现train()方法")


    def test(self):
        """Run testing/evaluation.

        Subclasses may implement this method."""
        raise NotImplementedError("子类必须实现test()方法")
