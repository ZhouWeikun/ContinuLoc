"""Expand one local entry file into self-contained published inference configs."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from continuloc.config.parser import _expand_selected_scene_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_contract(path, kind):
    """Paths may change on download; geometry and normalization must remain exact."""
    value = json.loads(Path(path).read_text())
    ignored = {'filepaths'} if kind == 'satellite' else {'uavimgs_dir', 'uavimgs_geocsv_path'}
    return {key: item for key, item in value.items() if key not in ignored}


def _relative(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Bundle assets must use relative paths without parent traversal: {value!r}')
    return root / path


def _local(root, value):
    return (root / Path(value).expanduser()).resolve()


def _selection(value, label):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'{label} must be a list of names')
    if len(set(value)) != len(value):
        raise ValueError(f'{label} contains duplicate names')
    return value


def expand_bundle(entry_path):
    """Validate everything first, then write derived inputs under the output root.

    Frozen assignments are never regenerated. Metadata byte hashes are rebound only
    after checking the CSV hash and all non-path metadata against frozen contracts.
    Neither the downloaded dataset nor the bundle assets are modified.
    """
    entry_path = Path(entry_path).resolve()
    entry = yaml.safe_load(entry_path.read_text())
    entry_keys = {'bundle', 'model_dir', 'manifest_dir', 'output_dir', 'experiments',
                  'scenes', 'hardware', 'eval_n_samples'}
    if not isinstance(entry, dict):
        raise ValueError('Bundle entry must be a YAML mapping')
    unknown = entry.keys() - entry_keys
    if unknown:
        names = ', '.join(sorted(map(str, unknown)))
        raise ValueError(f'Unsupported bundle entry keys: {names}. Set the dataset location with manifest_dir.')
    bundle_path = _local(entry_path.parent, entry['bundle'])
    root = bundle_path.parent
    bundle = yaml.safe_load(bundle_path.read_text())
    if bundle.get('schema_version') != 1:
        raise ValueError('Unsupported inference bundle schema')
    groups = _selection(entry['experiments'], 'experiments')
    if not groups or set(groups) - bundle['experiments'].keys():
        raise ValueError(f'Select experiments from {list(bundle["experiments"])}')
    selected_scenes = _selection(entry.get('scenes', []), 'scenes')
    available = {case['scene'] for group in groups for case in bundle['experiments'][group]}
    if set(selected_scenes) - available:
        raise ValueError(f'Unknown scenes for selected experiments: {set(selected_scenes) - available}')
    model_dir = _local(entry_path.parent, entry['model_dir'])
    output = _local(entry_path.parent, entry['output_dir'])
    manifest_dir = _local(entry_path.parent, entry['manifest_dir'])
    hardware = entry.get('hardware') or {}
    if hardware.keys() - {'gpu_ids', 'num_worker', 'satmaps_on_cpu'}:
        raise ValueError('hardware supports gpu_ids, num_worker and satmaps_on_cpu')
    limit = entry.get('eval_n_samples')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('eval_n_samples must be null or a positive integer')
    pending = []
    ids = set()
    for group in groups:
        for case in bundle['experiments'][group]:
            scene_name = case['scene']
            if selected_scenes and scene_name not in selected_scenes:
                continue
            case_id = f'{group}_{scene_name}'
            if case_id in ids or '/' in case_id or '\\' in case_id or case_id in ('.', '..'):
                raise ValueError(f'Invalid or duplicate case ID: {case_id}')
            ids.add(case_id)
            config = yaml.safe_load(_relative(root, case['config']).read_text())
            settings = dict(manifest_dir=str(manifest_dir), selected_scene_name=scene_name)
            # Resolve metadata without consulting the dataset's current split files.
            scene = _expand_selected_scene_config(settings)['scenes'][0]
            split = json.loads(_relative(root, case['split']).read_text())
            contract = json.loads(_relative(root, case['contract']).read_text())
            if split.get('scene') != scene_name or contract['scene'] != scene_name:
                raise ValueError(f'Bundle scene mismatch: {case_id}')
            if sha256(scene['p_uav_geocsv']) != split['source']['csv_sha256']:
                raise ValueError(f'CSV changed for frozen paper split: {scene_name}')
            if [Path(p).name for p in scene['satellite_filepaths']] != contract['satellite_filenames']:
                raise ValueError(f'Satellite source order changed: {scene_name}')
            for kind, key in [('satellite', 'p_satinfo_json'), ('uav', 'p_uavinfo_json')]:
                if metadata_contract(scene[key], kind) != contract[kind]:
                    raise ValueError(f'{kind} metadata geometry/normalization changed: {scene_name}')
                split['source'][f'{kind}_metadata_sha256'] = sha256(scene[key])
            exp = config['exp_setting']
            for stage in ('stage1', 'stage2'):
                path = _relative(model_dir, case[f'{stage}_weight']).absolute()
                if not path.is_file():
                    raise FileNotFoundError(path)
                exp[f'load_{stage}_ckpt'] = str(path)
            config['network_setting']['p_grid_config_yaml'] = str(_relative(root, case['grid']))
            exp['stage3_recall_cfg_yaml'] = str(_relative(root, case['thresholds']))
            config['scenes_setting'] = settings
            config['hardware_setting'].update(hardware)
            config['inference_setting']['hds_eval_n_samples'] = limit
            provenance = dict(entry=str(entry_path), entry_sha256=sha256(entry_path),
                              bundle=str(bundle_path), bundle_sha256=sha256(bundle_path),
                              case=case_id, frozen_split=str(_relative(root, case['split'])),
                              frozen_split_sha256=sha256(_relative(root, case['split'])),
                              contract_sha256=sha256(_relative(root, case['contract'])))
            pending.append((case_id, config, split, provenance))
    if not pending:
        raise ValueError('No selected inference cases')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    inputs = output / f'inputs_{stamp}'
    inputs.mkdir(parents=True, exist_ok=False)
    result = []
    for case_id, config, split, provenance in pending:
        target = inputs / case_id
        target.mkdir()
        (target / f'{split["scene"]}.json').write_text(json.dumps(split, indent=2, allow_nan=False)+'\n')
        (target / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        config['scenes_setting']['split_dir'] = str(target)
        config['exp_setting'].update(exp_name=case_id,
            stage3_analysis_export_root=str(output / f'{case_id}_{stamp}'))
        (target / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        result.append(copy.deepcopy(config))
    return result
