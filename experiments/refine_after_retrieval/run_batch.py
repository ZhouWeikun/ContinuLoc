"""Run YAML-configured retrieval refinement experiments sequentially."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_configs(base, text):
    plan = yaml.safe_load(text)
    if not isinstance(base, dict):
        raise ValueError('Base config must be a mapping')
    if not isinstance(plan, dict) or set(plan) != {'experiments'} or not isinstance(plan['experiments'], list):
        raise ValueError('Batch YAML must contain only an experiments list')
    configs, names = [], set()
    for experiment in plan['experiments']:
        if not isinstance(experiment, dict) or set(experiment) - {'name', 'enabled', 'overrides'}:
            raise ValueError('Each experiment accepts name, enabled, overrides only')
        name = experiment.get('name')
        if (not isinstance(name, str) or not name.strip() or name in names
                or name in ('.', '..') or any(c in name for c in '/\\')):
            raise ValueError(f'Invalid or duplicate experiment name: {name!r}')
        names.add(name)
        enabled = experiment.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: enabled must be a YAML boolean')
        if not enabled:
            continue
        overrides = experiment.get('overrides', {})
        if not isinstance(overrides, dict) or set(overrides) - set(base):
            raise ValueError(f'{name}: overrides contains unknown config keys')
        config = copy.deepcopy(base)
        config.update(overrides)
        config['name'] = name
        configs.append(config)
    return configs


def resolve_bundles(config):
    if not isinstance(config['retrieval_dir'], str) or not config['retrieval_dir'].strip():
        raise ValueError('retrieval_dir must be a nonempty path')
    root = project_path(config['retrieval_dir'])
    if (root / 'retrieval').is_dir():
        root = root / 'retrieval'
    split = config['split']
    if split not in ('val', 'test', 'all'):
        raise ValueError('split must be val, test or all')
    scenes = config['scenes']
    if not isinstance(scenes, list) or any(not isinstance(s, str) for s in scenes):
        raise ValueError('scenes must be a list of scene names')
    splits = ('val', 'test') if split == 'all' else (split,)
    bundles = sorted(path for part in splits for path in root.glob(f'*__{part}/retrieval_bundle.pt'))
    available = {p.parent.name.rsplit('__', 1)[0] for p in bundles}
    selected = set()
    for scene in scenes:
        # Prefer exact names; accept a unique suffix such as "01" for "visloc_01".
        matches = {scene} if scene in available else {
            name for name in available if name.rsplit('_', 1)[-1] == scene
        }
        if not matches:
            raise ValueError(
                f'Missing scene {scene!r} for split={split}; available: {sorted(available)}'
            )
        if len(matches) > 1:
            raise ValueError(f'Ambiguous scene {scene!r}; use a full name: {sorted(matches)}')
        selected.update(matches)
    bundles = [p for p in bundles if not scenes or p.parent.name.rsplit('__', 1)[0] in selected]
    if not bundles:
        raise FileNotFoundError(f'No retrieval bundles for split={split} under {root}')
    return bundles


def build_command(config, bundles):
    for key, minimum in (('topn_match', 1), ('min_inliers_for_refine', 4), ('limit_queries', 1)):
        value = config[key]
        if key == 'limit_queries' and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if not isinstance(config['save_intermediates'], bool):
        raise ValueError('save_intermediates must be a YAML boolean')
    for key in ('matcher_model', 'output_root'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    command = [sys.executable, '-u', str(HERE / 'run.py'),
               '--matcher-model', config['matcher_model'],
               '--topn-match', str(config['topn_match']),
               '--min-inliers-for-refine', str(config['min_inliers_for_refine']),
               '--output-root', str(project_path(config['output_root']) / config['name'])]
    for bundle in bundles:
        command.extend(['--bundle-path', str(bundle)])
    if config['weights_dir'] is not None:
        command.extend(['--weights-dir', str(project_path(config['weights_dir']))])
    if config['limit_queries'] is not None:
        command.extend(['--limit-queries', str(config['limit_queries'])])
    if config['save_intermediates']:
        command.append('--save-intermediates')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--batch-config', '--cfg-var-file', dest='batch_config', type=Path,
                        default=HERE / 'refine_batch_config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    configs = build_configs(yaml.safe_load(args.base_yaml.read_text()), args.batch_config.read_text())
    # Validate every task before starting any expensive matching work.
    tasks = [(config, resolve_bundles(config)) for config in configs]
    commands = [build_command(config, bundles) for config, bundles in tasks]
    for (config, bundles), command in zip(tasks, commands):
        print(f'{config["name"]}: {len(bundles)} scene/split bundles', flush=True)
        print(shlex.join(command), flush=True)
    if args.dry_run or not tasks:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_dir = ROOT / 'logs' / 'refine_after_retrieval_batch' / stamp
    log_dir.mkdir(parents=True, exist_ok=False)
    for (config, bundles), command in zip(tasks, commands):
        task_log_dir = log_dir / config['name']
        task_log_dir.mkdir()
        stem = task_log_dir / 'run'
        stem.with_suffix('.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        stem.with_suffix('.json').write_text(json.dumps({
            'command': command, 'bundles': [str(p) for p in bundles],
        }, indent=2))
        with stem.with_suffix('.log').open('x') as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code:
            raise SystemExit(code)
    print(f'Completed batch. Logs: {log_dir}', flush=True)


if __name__ == '__main__':
    main()
