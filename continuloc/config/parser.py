#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified configuration parsing for command-line arguments, YAML files, and multiple scenes."""

import argparse
import yaml
import os
import copy
import json
from pathlib import Path


def _parse_boolean(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def print_config_summary(opt, header="最终配置:"):
    """Print a summary of the current options."""
    scenes_setting = getattr(opt, 'scenes_setting', None)
    if isinstance(scenes_setting, dict):
        scenes = scenes_setting.get('scenes', [])
        print(f"\n{'='*60}")
        print(f"场景配置: {'多场景模式' if len(scenes) > 1 else '单场景模式'} ({len(scenes)}个场景)")
        for i, scene in enumerate(scenes):
            print(f"  场景{i+1}: {scene.get('name', f'scene_{i+1}')}")
        if len(scenes) > 1 and 'sampling_strategy' in scenes_setting:
            print(f"  采样策略: {scenes_setting['sampling_strategy']}")
        print(f"{'='*60}\n")

    print(header)
    for group_name, keys in getattr(opt, 'group_dict', {}).items():
        print(f"  [{group_name}]")
        for key in keys:
            if hasattr(opt, key):
                print(f"    {key}: {getattr(opt, key)}")
    if hasattr(opt, 'scenes_setting'):
        print(f"  [scenes_setting]: {opt.scenes_setting}")

    opt._config_summary_printed = True


def _resolve_manifest_location(settings):
    """Accept a directory, the explicit file key, and the legacy file alias."""
    directory = settings.get('manifest_dir')
    directory = Path(directory).expanduser().resolve() if directory else None
    explicit = []
    for key in ('p_dataset_manifest', 'dataset_manifest'):
        if settings.get(key):
            path = Path(settings[key]).expanduser()
            explicit.append((key, ((directory / path) if directory else path).resolve()))
    if len({path for _, path in explicit}) > 1:
        raise ValueError('p_dataset_manifest and dataset_manifest refer to different files')
    if explicit:
        manifest_path = explicit[0][1]
        if directory is not None and manifest_path.parent != directory:
            raise ValueError('manifest_dir must be the parent of the explicit dataset manifest')
    elif directory is not None:
        manifest_path = directory / 'dataset_manifest.json'
    else:
        raise ValueError('Specify manifest_dir, p_dataset_manifest, or dataset_manifest')
    return manifest_path


def _expand_manifest_scenes(settings, require_splits=False):
    """Resolve schema-v2 scenes and discover per-scene JSONs under manifest_dir/splits."""
    expanded = copy.deepcopy(settings)
    manifest_path = _resolve_manifest_location(settings)
    split_dir = (manifest_path.parent / Path(settings.get('split_dir') or 'splits').expanduser()).resolve()
    strict_splits = require_splits or bool(settings.get('split_dir'))
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    schema = manifest.get('schema_version')
    if schema != 2:
        raise ValueError(f'Unsupported manifest schema_version: {schema}')
    root_keys = ('visloc_dir', 'wingta_dir')
    registry = manifest['scene_registry']
    dataset = manifest['dataset_name']
    if settings.get('dataset_name', dataset) != dataset:
        raise ValueError("dataset_name does not match dataset_manifest")
    single = settings.get('selected_scene_name')
    multiple = settings.get('selected_scene_names')
    if single is not None and multiple is not None:
        raise ValueError("Specify selected_scene_name OR selected_scene_names, not both")
    requested = [single] if single is not None else multiple
    if requested is None:
        requested = manifest.get('default_scene_names', [])
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError("selected_scene_names must be a nonempty list")
    names, scenes = [], []
    def resolve_path(value, directory=False):
        path = Path(value).expanduser()
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Scene {name}: expected a path relative to {root_key}: {value!r}')
        path = (scene_root / path).resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(f"Manifest path does not exist: {path}")
        return str(path)
    for item in requested:
        name = str(item).strip()
        if name.isdigit():
            prefix = "visloc" if f"visloc_{int(name):02d}" in registry else dataset
            name = f"{prefix}_{int(name):02d}"
        if name not in registry:
            raise ValueError(f"Unknown scene {item!r}; available: {', '.join(registry)}")
        if name in names:
            raise ValueError(f"Duplicate scene selection: {name}")
        scene = copy.deepcopy(registry[name])
        if scene.get('status') != 'prepared':
            raise ValueError(f"Scene {name} is not prepared")
        scene['name'] = name
        root_key = scene.get('dataset_root')
        if root_key not in root_keys:
            raise ValueError(f'Scene {name}: dataset_root must be one of {root_keys}')
        root_value = settings.get(root_key, manifest.get(root_key))
        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError(f'Scene {name}: missing dataset root {root_key}')
        root_path = Path(root_value).expanduser()
        scene_root = (manifest_path.parent / root_path).resolve()
        for key in ('p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'):
            scene[key] = resolve_path(scene[key])
        if scene.get('global_negative_sampling_grid') is not None:
            grid = scene['global_negative_sampling_grid']
            for key in ('p_grid_npz', 'p_grid_metadata_json', 'p_grid_tif'):
                if key in grid:
                    grid[key] = resolve_path(grid[key])
        scene['uavimgs_dir'] = resolve_path(scene['uavimgs_dir'], directory=True)
        if not scene.get('satellite_filepaths'):
            raise ValueError(f"Scene {name} has no satellite_filepaths")
        scene['satellite_filepaths'] = [resolve_path(v) for v in scene['satellite_filepaths']]
        split_path = split_dir / f'{name}.json'
        if strict_splits and not split_path.is_file():
            raise FileNotFoundError(f'Fixed split does not exist: {split_path}')
        if split_path.is_file():
            scene['p_split_json'] = str(split_path)
        names.append(name)
        scenes.append(scene)
    expanded.update(dataset_name=dataset, manifest_dir=str(manifest_path.parent),
                    p_dataset_manifest=str(manifest_path), dataset_manifest=str(manifest_path), scenes=scenes)
    if single is not None:
        expanded['selected_scene_name'] = names[0]
    else:
        expanded['selected_scene_names'] = names
    return expanded


def _expand_selected_scene_config(scenes_setting, require_splits=False):
    if not isinstance(scenes_setting, dict):
        return scenes_setting

    if any(scenes_setting.get(key) for key in ('manifest_dir', 'p_dataset_manifest', 'dataset_manifest')):
        return _expand_manifest_scenes(scenes_setting, require_splits=require_splits)

    if scenes_setting.get('selected_scene_names') is not None:
        raise ValueError('selected_scene_names requires manifest_dir or a dataset manifest file')

    scene_registry = scenes_setting.get('scene_registry', None)
    if not isinstance(scene_registry, dict) or not scene_registry:
        return scenes_setting

    selected_scene_name = scenes_setting.get('selected_scene_name', None)
    if not selected_scene_name:
        raise ValueError("scenes_setting.scene_registry is provided, but selected_scene_name is missing.")
    if selected_scene_name not in scene_registry:
        available = ", ".join(sorted(scene_registry.keys()))
        raise KeyError(
            f"selected_scene_name '{selected_scene_name}' not found in scene_registry. "
            f"Available scenes: {available}"
        )

    selected_scene = copy.deepcopy(scene_registry[selected_scene_name])
    if not isinstance(selected_scene, dict):
        raise TypeError(
            f"scene_registry['{selected_scene_name}'] must be a dict, "
            f"got {type(selected_scene).__name__}"
        )
    selected_scene.setdefault('name', selected_scene_name)

    expanded = copy.deepcopy(scenes_setting)
    expanded['selected_scene_name'] = selected_scene_name
    if selected_scene.get('dataset_name'):
        expanded['dataset_name'] = selected_scene['dataset_name']
    expanded['scenes'] = [selected_scene]
    return expanded


def get_parse(print_summary=True):
    """Parse CLI arguments and YAML configuration.

    Precedence: CLI arguments > saved run YAML > base YAML > defaults."""
    parser = argparse.ArgumentParser(description='Training')

    # ==================== Core configuration ====================
    parser.add_argument('--p_yaml',
                        default='configs/visual_encoder_continuscenes.yaml',
                        type=str, help='YAML配置文件路径 (通常指向opts.yaml或基础配置)')

    # Experiment settings
    parser.add_argument('--dir2save_log', default='gen_fm_exps/logs', type=str, help='日志与TensorBoard保存根目录')
    parser.add_argument('--dir2save_ckpt', default='gen_fm_exps/ckpts', type=str, help='Checkpoint保存根目录')
    parser.add_argument('--exps_dir', default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument('--exp_name', default='debug', type=str, help='实验名称')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='是否使用tensorboard')

    # Checkpoint settings (dictionary or string; overridden by YAML)
    parser.add_argument('--load2test', default="", type=str, help='测试时加载的checkpoint')
    parser.add_argument('--load2train', default="", type=str, help='继续训练时加载的checkpoint')

    # Stage checkpoints for multi-stage training
    parser.add_argument('--load_stage1_ckpt', default="", type=str, help='Stage 1的checkpoint路径')
    parser.add_argument('--load_stage2_ckpt', default="", type=str, help='Stage 2的checkpoint路径')
    parser.add_argument('--stage3_analysis_export_root', default="", type=str, help='Stage 3测试分析结果导出目录')
    parser.add_argument('--stage3_recall_cfg', default="per_scene", type=str, help='Stage 3 recall阈值配置名；per_scene表示按场景映射')
    parser.add_argument('--stage3_recall_cfg_yaml', default="configs/pose_inference_recall_thresholds.yaml", type=str, help='Stage 3 recall阈值配置YAML')
    parser.add_argument(
        '--stage3_print_progressive_recall',
        default=False,
        type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y'},
        help='是否打印 Stage 3 progressive recall 报告；默认关闭，仅保存结果',
    )
    parser.add_argument('--stage3_basin_enable', default=False, type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y'}, help='是否在Stage 3 test中运行CMA-ES吸引盆分析')
    parser.add_argument('--stage3_basin_output_root', default="", type=str, help='Stage 3吸引盆分析输出目录；为空时使用stage3_analysis_export_root/stage3_basin')
    parser.add_argument('--stage3_basin_n_samples', default=256, type=int, help='Stage 3吸引盆分析测试样本数')
    parser.add_argument('--stage3_basin_query_ids', default="", type=str, help='逗号分隔的query索引；非空时覆盖n_samples')
    parser.add_argument('--stage3_basin_num_particles', default=128, type=int, help='每个query的CMA-ES population size')
    parser.add_argument('--stage3_basin_query_batch_size', default=8, type=int, help='吸引盆分析每批query数量')
    parser.add_argument('--stage3_basin_sample_radius_rc', default="5.0,5.0", type=str, help='围绕GT采样的nr,nc半径')
    parser.add_argument('--stage3_basin_sample_radius_rot_deg', default=0.0, type=float, help='围绕GT采样的旋转半径，单位degree')
    parser.add_argument('--stage3_basin_sample_radius_scale_ratio', default=0.0, type=float, help='围绕GT采样的相对scale半径')
    parser.add_argument('--stage3_basin_fix_rot', default=True, type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'}, help='吸引盆分析中是否固定rot维度')
    parser.add_argument('--stage3_basin_fix_scale', default=True, type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'}, help='吸引盆分析中是否固定scale维度')
    parser.add_argument('--stage3_basin_seed', default=None, help='吸引盆分析随机种子；None/null/空字符串表示不固定')
    parser.add_argument('--stage3_basin_chunk_size', default=8192, type=int, help='吸引盆分析评分chunk size')
    parser.add_argument('--stage3_basin_cma_sigma0', default=0.2, type=float, help='吸引盆分析CMA-ES初始sigma')
    parser.add_argument('--stage3_basin_cma_popsize', default=16, type=int, help='兼容保留；当前单种群实现使用stage3_basin_num_particles作为population size')
    parser.add_argument('--stage3_basin_cma_iters', default=20, type=int, help='吸引盆分析CMA-ES迭代轮数')
    parser.add_argument('--stage3_basin_cma_variant', default="Sep-CMA", type=str, help='吸引盆分析CMA-ES后端: CMA或Sep-CMA')
    parser.add_argument('--stage3_basin_cma_prob_mode', default="product", type=str, help='吸引盆分析CMA-ES评分模式: ingp/projector/product')
    parser.add_argument('--stage3_basin_cma_enable_early_stop', default=True, type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'}, help='吸引盆分析CMA-ES是否早停')
    parser.add_argument('--stage3_basin_cma_early_stop_patience', default=5, type=int, help='吸引盆分析CMA-ES早停patience')
    parser.add_argument('--stage3_basin_optimizer_backend', default="cma_es", type=str, help='吸引盆分析优化后端: cma_es或stage1_5')
    parser.add_argument('--stage3_basin_stage1_5_iters', default=None, type=int, help='stage1_5后端迭代轮数；None时复用stage3_basin_cma_iters')
    parser.add_argument('--stage3_basin_stage1_5_elite_ratio', default=0.125, type=float, help='stage1_5后端每轮elite比例')
    parser.add_argument('--stage3_basin_stage1_5_radius_decay', default=1.0, type=float, help='stage1_5后端每轮采样半径衰减系数')
    parser.add_argument('--stage3_basin_stage1_5_move_stand', default="elite_sum", type=str, help='stage1_5后端中心移动策略: best或elite_sum')
    parser.add_argument('--stage3_basin_save_particles', default=False, type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y'}, help='是否保存per-particle明细')
    parser.add_argument('--stage3_basin_use_train_uav', default=False, type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y'}, help='吸引盆分析是否使用train UAV split')
    parser.add_argument('--stage3_basin_shuffle', default=False, type=lambda x: str(x).strip().lower() in {'1', 'true', 'yes', 'y'}, help='吸引盆分析是否shuffle测试样本')
    parser.add_argument('--stage3_basin_progress', default=True, type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'}, help='吸引盆分析是否显示进度条')
    parser.add_argument('--inherit_stage1_yaml', default="", type=str, help='Stage 2初始化时继承Stage 1配置参数的opts/base yaml路径')
    parser.add_argument('--inherit_stage1_scope', default='network,data', type=str, help='Stage 2从Stage 1 YAML继承的范围: network,data,scenes,hardware，可用逗号分隔')
    parser.add_argument('--inherit_stage2_yaml', default="", type=str, help='Stage 3初始化时继承Stage 2网络结构参数的opts/base yaml路径')
    parser.add_argument('--inherit_stage2_scope', default='network', type=str, help='Stage 3从Stage 2 YAML继承的范围: network,data,scenes,hardware，可用逗号分隔')
    parser.add_argument('--selected_scene_names', nargs='+', default=None, help='覆盖 manifest 场景列表，支持名称或编号')
    parser.add_argument('--selected_scene_name', default="", type=str, help='覆盖 scenes_setting.selected_scene_name')
    parser.add_argument(
        '--inherit_stage1_fm_stage2',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='Stage 3是否通过Stage 2的配置继续继承Stage 1相关默认参数',
    )
    parser.add_argument(
        '--inherit_stage1_fm_stage2_scope',
        default="",
        type=str,
        help='Stage 3经由Stage 2继续继承Stage 1时使用的scope；为空时回退到Stage 2记录的 inherit_stage1_scope',
    )

    # Hardware settings
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs, 例如: 0 或 0,1,2')
    parser.add_argument('--num_worker', default=8, type=int, help='DataLoader worker数量')
    parser.add_argument(
        '--satmaps_on_cpu',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否将 SatDataset 的整图缓存与裁图留在 CPU 上',
    )
    parser.add_argument('--autocast', action='store_true', default=False, help='是否使用混合精度训练')

    # Network settings
    parser.add_argument('--backbone', default="dinov3", type=str,
                        help='Backbone类型: ViTB-224, dinov2, dinov3等')
    parser.add_argument(
        '--freeze_backbone',
        default=True,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否冻结视觉backbone；False时允许backbone参与训练',
    )
    parser.add_argument('--aggregator_type', default='salad', type=str,
                        help='聚合器类型: salad, g2m, g2m_scalar_p, g2m_channelwise_p, gem, fsra, lpn, netvlad')

    # Training settings
    parser.add_argument('--num_epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--scale_filter_mode', default=None, choices=['none', 'percentile', 'mad'],
                        help='尺度过滤：none 不过滤；percentile 严格 P2/P99；mad 复现 master 通用 MAD 3.5。未配置时兼容旧行为。')
    parser.add_argument('--enable_edge_filter', type=_parse_boolean, default=None,
                        help='是否启用卫星采样边缘过滤；JSON 模式下用于选择已保存的过滤方案')
    parser.add_argument('--split_source', default='generated', choices=['generated', 'csv', 'json'], help='按比例生成划分，读取 CSV 列，或读取固定 JSON（比例由 JSON 决定）')
    parser.add_argument('--split_val_ratio', default=0.0, type=float, help='验证集占全部参与划分样本的比例；从原测试集合中划出，默认关闭')
    parser.add_argument('--split_train_ratio', default=0.9, type=float, help='UAV 数据集训练集比例')
    parser.add_argument(
        '--split_mode',
        default='segment',
        type=str,
        choices=('segment', 'interval', 'random'),
        help='UAV 数据集切分模式: segment=前段训练后段测试, interval=按固定间隔交错切分, random=固定随机种子划分',
    )
    parser.add_argument('--n_global_negatives_per_batch', default=384, type=int,
                        help='每个训练 batch 的全局随机负样本总数；0 表示不额外采样')
    parser.add_argument(
        '--reject_sampling', '--reject_sampleing',
        dest='reject_sampling',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否让全局负样本避开当前 batch 的全部 query 邻域',
    )
    parser.add_argument(
        '--sat_as_query',
        default=False,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='是否将sat image也作为query构造配对样本',
    )
    parser.add_argument(
        '--pair_alignment_mode',
        default='full_4d',
        type=str,
        choices=('full_4d', 'xy_only'),
        help='query-positive 配对对齐方式: full_4d=xy/rot/scale对齐, xy_only=仅xy对齐且sat positive固定rot=0、scale=train-mean',
    )
    parser.add_argument(
        '--sat_scale_sampling_mode',
        default='empirical',
        type=str,
        choices=('empirical', 'uniform_linear', 'uniform_log'),
        help='Stage 2卫星裁剪尺度采样: empirical=按UAV经验分布, uniform_linear=raw scale均匀, uniform_log=log(scale)均匀',
    )
    parser.add_argument(
        '--sat_scale_jitter_enabled',
        default=True,
        type=lambda x: str(x).strip().lower() not in {'0', 'false', 'no', 'n'},
        help='Stage 2卫星裁剪尺度是否叠加随机扰动',
    )
    parser.add_argument(
        '--sat_scale_jitter_half_span_ratio',
        default=0.05,
        type=float,
        help='尺度扰动半径占完整尺度跨度的比例；0.05表示加上±5%%*(scale_max-scale_min)',
    )

    parser.add_argument(
        '--use_uavimgs',
        # Keep an unset sentinel so explicit CLI false overrides YAML true.
        default=None,
        type=_parse_boolean,
        help='Neural-field training: false = satellite crops only (default); true = satellite + UAV images',
    )

    # Grid settings
    parser.add_argument('--freeze_grid', action='store_true', default=False,
                        help='是否冻结Grid（Stage 3中使用）')
    parser.add_argument('--p_grid_config_yaml', default=None, type=str,
                        help='神经场网格配置路径；默认使用 YAML 或检查点记录的结构')

    # ==================== Legacy single-scene options ====================
    parser.add_argument('--p_satinfo_json', default='', type=str, help='卫星图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uavinfo_json', default='', type=str, help='UAV图信息JSON路径（向后兼容）')
    parser.add_argument('--p_uav_geocsv', default='', type=str, help='UAV地理信息CSV路径（向后兼容）')
    parser.add_argument('--dataset_name', default='default', type=str, help='数据集名称（向后兼容）')

    # --- 1. Parse the initial command-line arguments ---
    opt = parser.parse_args()
    default_args = parser.parse_args(args=[]) # Save defaults for comparison.
    
    # Resolve the absolute path of this file (parser.py).
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Ascend two levels from this directory to the project root.
    project_root = os.path.dirname(os.path.dirname(current_file_dir))

    # --- 2. Define configuration dictionary merging ---
    def merge_configs(base_dict, override_dict):
        """Recursively merge dictionaries, giving override_dict precedence over base_dict."""
        for k, v in override_dict.items():
            if isinstance(v, dict) and isinstance(base_dict.get(k), dict):
                base_dict[k] = merge_configs(base_dict[k], v)
            else:
                base_dict[k] = v
        return base_dict

    # --- 3. Load the base configuration and saved run configuration ---
    config_from_base_yaml = {}
    config_from_exp_log = {}
    
    # Load the requested YAML, which may be a base configuration or a saved run.
    if opt.p_yaml:
        first_yaml_path_abs = os.path.join(project_root, opt.p_yaml)
        if os.path.exists(first_yaml_path_abs):
            print(f"从命令行指定YAML文件 '{first_yaml_path_abs}' 加载配置...")
            with open(first_yaml_path_abs, 'r', encoding='utf-8') as f:
                temp_config = yaml.safe_load(f)
            
            # A p_yaml key identifies a saved run configuration.
            if 'p_yaml' in temp_config or ('exp_setting' in temp_config and 'p_yaml' in temp_config['exp_setting']):
                # This YAML describes a saved run.
                config_from_exp_log = temp_config
                base_yaml_path_rel = config_from_exp_log.get('p_yaml') or (config_from_exp_log.get('exp_setting', {})).get('p_yaml')

                # Load the base configuration to supply additional defaults.
                # The base file may be absent if the saved run contains a complete configuration.
                if base_yaml_path_rel:
                    base_yaml_path_abs = os.path.join(project_root, base_yaml_path_rel)
                    if os.path.exists(base_yaml_path_abs):
                        print(f"从实验日志中解析到基础配置文件 '{base_yaml_path_abs}' 加载配置（作为默认值补充）...")
                        with open(base_yaml_path_abs, 'r', encoding='utf-8') as f:
                            config_from_base_yaml = yaml.safe_load(f)
                    else:
                        print(f"提示: 基础配置文件 '{base_yaml_path_abs}' 不存在，将使用实验日志中的完整配置。")
                else:
                    print(f"提示: 实验日志 '{first_yaml_path_abs}' 中未指定基础配置文件，将使用实验日志中的完整配置。")
            else:
                # This is a base configuration file.
                config_from_base_yaml = temp_config
                print(f"检测到 '{first_yaml_path_abs}' 是基础配置文件。")
        else:
            print(f"警告: 命令行指定的YAML文件 '{first_yaml_path_abs}' 不存在。")

    # Load the default base file if none was specified or the saved run lacks p_yaml.
    if not config_from_base_yaml:
        default_base_yaml_path_rel = default_args.p_yaml
        default_base_yaml_path_abs = os.path.join(project_root, default_base_yaml_path_rel)
        if os.path.exists(default_base_yaml_path_abs):
            print(f"加载默认基础配置文件 '{default_base_yaml_path_abs}' 加载配置...")
            with open(default_base_yaml_path_abs, 'r', encoding='utf-8') as f:
                config_from_base_yaml = yaml.safe_load(f)
        else:
            print(f"警告: 默认基础配置文件 '{default_base_yaml_path_abs}' 不存在。")

    # --- 4. Merge configurations (base, then saved run) ---
    final_merged_config = merge_configs(config_from_base_yaml, config_from_exp_log)

    # --- 5. Apply merged settings to opt, respecting CLI arguments ---
    # Apply the merged YAML configuration to opt first.
    for section, params in final_merged_config.items():
        if section == 'scenes_setting':
            setattr(opt, 'scenes_setting', copy.deepcopy(params))
            continue

        if isinstance(params, dict):
            if section == 'data_setting':
                removed = {'n_neg_per_query', 'add_random_satimg_negs', 'reject_batch_aware'} & params.keys()
                if removed:
                    raise ValueError(f'Obsolete negative-sampling settings: {sorted(removed)}. Use n_global_negatives_per_batch and reject_sampling (all queries in the batch).')
            if section == 'data_setting' and 'reject_sampleing' in params and 'reject_sampling' not in params:
                params = dict(params)
                params['reject_sampling'] = params['reject_sampleing']
            for key, value in params.items():
                # YAML may override a CLI option only when it still has its default value.
                # YAML-only keys may already exist from an earlier section.
                # They have no argparse default and remain overridable by YAML.
                if not hasattr(default_args, key) or getattr(opt, key) == getattr(default_args, key):
                    setattr(opt, key, value)
        else: # Top-level options
            if not hasattr(default_args, section) or getattr(opt, section) == getattr(default_args, section):
                setattr(opt, section, params)
    
    if opt.use_uavimgs is None:
        opt.use_uavimgs = False
    if not isinstance(opt.use_uavimgs, bool):
        raise ValueError("data_setting.use_uavimgs must be a YAML boolean (true/false).")

    if opt.scale_filter_mode is not None and opt.scale_filter_mode not in {"none", "percentile", "mad"}:
        raise ValueError("scale_filter_mode must be none, percentile, or mad.")
    if opt.enable_edge_filter is None:
        opt.enable_edge_filter = True
    if not isinstance(opt.enable_edge_filter, bool):
        raise ValueError('enable_edge_filter must be a YAML boolean')
    if opt.split_source == 'json':
        from continuloc.datasets.fixed_splits import select_filter_profile
        select_filter_profile(opt.enable_edge_filter, opt.scale_filter_mode)
        if opt.split_val_ratio != 0:
            raise ValueError('Fixed train/test JSON requires split_val_ratio=0')

    # --- 6. Validate and populate scene settings (legacy compatibility) ---
    if not hasattr(opt, 'scenes_setting') or not opt.scenes_setting:
        print("警告：未找到 scenes_setting，将使用命令行参数作为单场景配置")
        # Fall back to legacy single-scene settings when YAML lacks scenes_setting.
        opt.scenes_setting = {
            'sampling_strategy': 'round_robin',
            'scenes': [{
                'name': getattr(opt, 'dataset_name', 'default'),
                'p_satinfo_json': opt.p_satinfo_json,
                'p_uavinfo_json': opt.p_uavinfo_json,
                'p_uav_geocsv': opt.p_uav_geocsv,
                'weight': 1.0
            }]
        }
    else:
        if getattr(opt, 'selected_scene_name', '') and getattr(opt, 'selected_scene_names', None):
            raise ValueError('Use only one scene selection CLI option')
        if getattr(opt, 'selected_scene_names', None):
            opt.scenes_setting = copy.deepcopy(opt.scenes_setting)
            opt.scenes_setting.pop('selected_scene_name', None)
            opt.scenes_setting['selected_scene_names'] = opt.selected_scene_names
        if getattr(opt, 'selected_scene_name', ''):
            if not isinstance(opt.scenes_setting, dict):
                raise TypeError("selected_scene_name override requires scenes_setting to be a dict")
            opt.scenes_setting = copy.deepcopy(opt.scenes_setting)
            opt.scenes_setting.pop('selected_scene_names', None)
            opt.scenes_setting['selected_scene_name'] = opt.selected_scene_name
        opt.scenes_setting = _expand_selected_scene_config(
            opt.scenes_setting, require_splits=opt.split_source == 'json')

    legacy_exps_dir = getattr(opt, 'exps_dir', None)
    if not getattr(opt, 'dir2save_log', None):
        opt.dir2save_log = default_args.dir2save_log
    if not getattr(opt, 'dir2save_ckpt', None):
        opt.dir2save_ckpt = default_args.dir2save_ckpt
    if legacy_exps_dir:
        if opt.dir2save_log == default_args.dir2save_log:
            opt.dir2save_log = legacy_exps_dir
        if opt.dir2save_ckpt == default_args.dir2save_ckpt:
            opt.dir2save_ckpt = legacy_exps_dir
        print("提示: `exps_dir` 已废弃，请改用 `dir2save_log` 和 `dir2save_ckpt`。")

    # Output roots are relative to this checkout, independent of the launch cwd.
    project_root = Path(__file__).resolve().parents[2]
    for key in ('dir2save_log', 'dir2save_ckpt'):
        path = Path(getattr(opt, key)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        setattr(opt, key, str(path.resolve()))

    # Map opt.exps_dir to the log root for legacy callers that still read it directly.
    opt.exps_dir = opt.dir2save_log

    backbone_config = getattr(opt, 'backbone_config', {})
    if backbone_config is None:
        backbone_config = {}
    if not isinstance(backbone_config, dict):
        raise TypeError(f"backbone_config must be a dict, got {type(backbone_config).__name__}")
    opt.backbone_config = dict(backbone_config)

    aggregator_config = getattr(opt, 'aggregator_config', {})
    if aggregator_config is None:
        aggregator_config = {}
    if not isinstance(aggregator_config, dict):
        raise TypeError(f"aggregator_config must be a dict, got {type(aggregator_config).__name__}")
    opt.aggregator_config = dict(aggregator_config)
    if hasattr(opt, 'reject_sampleing'):
        opt.reject_sampling = bool(getattr(opt, 'reject_sampleing'))

    adapter_config = getattr(opt, 'adapter_config', {})
    if adapter_config is None:
        adapter_config = {}
    if not isinstance(adapter_config, dict):
        raise TypeError(f"adapter_config must be a dict, got {type(adapter_config).__name__}")
    opt.adapter_config = dict(adapter_config)

    hash_lod_aggregator = getattr(opt, 'hash_lod_aggregator', {})
    if hash_lod_aggregator is None:
        hash_lod_aggregator = {}
    if not isinstance(hash_lod_aggregator, dict):
        raise TypeError(f"hash_lod_aggregator must be a dict, got {type(hash_lod_aggregator).__name__}")
    opt.hash_lod_aggregator = dict(hash_lod_aggregator)

    # --- 7. Organize options into group_dict ---
    group_info = {
        'exp_setting': ['p_yaml', 'exp_name', 'dir2save_log', 'dir2save_ckpt', 'load2train', 'load2test',
                        'load_stage1_ckpt', 'load_stage2_ckpt',
                        'stage3_analysis_export_root',
                        'stage3_recall_cfg', 'stage3_recall_cfg_yaml',
                        'stage3_basin_enable', 'stage3_basin_output_root',
                        'stage3_basin_n_samples', 'stage3_basin_query_ids',
                        'stage3_basin_num_particles', 'stage3_basin_query_batch_size',
                        'stage3_basin_sample_radius_rc',
                        'stage3_basin_sample_radius_rot_deg',
                        'stage3_basin_sample_radius_scale_ratio',
                        'stage3_basin_fix_rot', 'stage3_basin_fix_scale',
                        'stage3_basin_seed', 'stage3_basin_chunk_size',
                        'stage3_basin_cma_sigma0', 'stage3_basin_cma_popsize',
                        'stage3_basin_cma_iters', 'stage3_basin_cma_variant',
                        'stage3_basin_cma_prob_mode',
                        'stage3_basin_cma_enable_early_stop',
                        'stage3_basin_cma_early_stop_patience',
                        'stage3_basin_optimizer_backend',
                        'stage3_basin_stage1_5_iters',
                        'stage3_basin_stage1_5_elite_ratio',
                        'stage3_basin_stage1_5_radius_decay',
                        'stage3_basin_stage1_5_move_stand',
                        'stage3_basin_save_particles',
                        'stage3_basin_use_train_uav',
                        'stage3_basin_shuffle', 'stage3_basin_progress',
                        'inherit_stage1_yaml', 'inherit_stage1_scope',
                        'inherit_stage2_yaml', 'inherit_stage2_scope',
                        'selected_scene_name', 'selected_scene_names',
                        'inherit_stage1_fm_stage2', 'inherit_stage1_fm_stage2_scope',
                        'tensorboard',
                        'save_freq', 'val', 'val_freq',
                        'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv'],
        'data_setting': [
            'enable_edge_filter',
            'scale_filter_mode',
            'pad_mode',
            'imgsize2net',
            'satimgsize2crop',
            'n_rand2sample_per_pos',
            'split_train_ratio',
            'split_val_ratio',
            'split_source',
            'split_mode',
            'n_global_negatives_per_batch',
            'reject_sampling',
            'sat_as_query',
            'use_uavimgs',
            'pair_alignment_mode',
            'sat_scale_sampling_mode',
            'sat_scale_jitter_enabled',
            'sat_scale_jitter_half_span_ratio',
        ],
        'hardware_setting': ['gpu_ids', 'num_worker', 'satmaps_on_cpu', 'autocast',
                            'batchsize_sat', 'batchsize_uav', 'batchsize_uav_test'],
        'network_setting': [
            'backbone',
            'freeze_backbone',
            'backbone_config',
            'aggregator_type',
            'aggregator_config',
            'adapter_config',
            'freeze_grid',
            'p_grid_config_yaml',
            'hash_lod_aggregator',
            'posenc_multires_rc',
            'posenc_multires_rot',
            'posenc_multires_scale',
            'grid_mlp_hidden_dim',
            'grid_mlp_num_blocks',
            'grid_mlp_arch',
            'grid_mlp_use_coord_condition',
            'apr_target_mode',
            'apr_mlp_hidden_dim',
            'apr_mlp_num_layers',
            'apr_mlp_norm',
            'apr_mlp_dropout',
            'apr_mlp_activation',
            'apr_output_tanh',
            'apr_normalize_input_feat',
            'apr_eval_clamp',
            'apr_require_stage1_ckpt',
        ],
        'learning_setting': [
            'num_epochs',
            'loss_type',
            'infonce_temperature',
            'infonce_negative_mode',
            'infonce_learnable_temperature',
            'infonce_gate',
            'infonce_metric',
            'infonce_normalize',
            'triplet_alpha',
            'triplet_margin',
            'triplet_gate',
            'triplet_enable_row_filter',
            'triplet_learnable_alpha',
            'triplet_metric',
            'triplet_reduction',

            'apr_optimizer',
            'apr_lr',
            'apr_weight_decay',
            'apr_loss_weight_nrc',
            'apr_loss_weight_rot',
            'apr_loss_weight_scale',
            'apr_max_eval_batches',
        ],
        'inference_setting': [
            'hds_stage1_overlap',
            'hds_stage1_n_rot',
            'hds_stage1_n_scale',
            'hds_stage1_scale_spacing',
            'hds_stage1_gallery_feature_source',
            'hds_stage1_visual_gallery_batch_size',
            'hds_stage1_visual_score_mode',
            'hds_stage1_visual_softmax_temperature',
            'hds_stage1_top_ratio_rho0',
            'hds_stage1_max_anchors_k0',
            'hds_stage2_rounds_t',
            'hds_stage2_particles_m',
            'hds_stage2_initial_radius_eta',
            'hds_stage2_radius_decay_alpha',
            'hds_stage2_elite_ratio_rho',
            'hds_stage2_population_survival_ratio_rho_pop',
            'hds_stage2_enable_scale_sampling',
            'hds_stage3_max_modes_k2',
            'hds_stage3_cma_variant',
            'hds_stage3_cma_sigma_factor',
            'hds_stage3_cma_popsize',
            'hds_stage3_cma_iters',
            'hds_stage3_cma_early_stop_patience',
            'hds_stage3_optimize_scale',
            'hds_score_chunk_size',
            'hds_eval_n_samples',
            'hds_scale_select_mode',
            'hds_scale_select_scope',
        ],
    }
    opt.group_dict = group_info
    opt._config_summary_printed = False

    # --- 8. Print the final configuration and return ---
    if print_summary:
        print_config_summary(opt)

    return opt
