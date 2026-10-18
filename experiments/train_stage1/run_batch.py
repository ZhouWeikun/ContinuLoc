"""Run sequential Stage 1 experiments described by a YAML experiment list."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def build_configs(base, text):
    plan = yaml.safe_load(text)
    if not isinstance(plan, dict) or set(plan) != {'experiments'}:
        raise ValueError('Batch YAML must contain only an experiments list')
    if not isinstance(plan['experiments'], list):
        raise ValueError('experiments must be a list')
    configs, names = [], set()
    for experiment in plan['experiments']:
        if not isinstance(experiment, dict) or set(experiment) - {'name', 'enabled', 'base_yaml', 'overrides'}:
            raise ValueError('Each experiment accepts name, enabled, base_yaml, overrides only')
        name = experiment.get('name')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f'Empty or duplicate experiment name: {name!r}')
        names.add(name)
        enabled = experiment.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: enabled must be a YAML boolean (true/false)')
        if not enabled:
            continue
        overrides = experiment.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError(f'{name}: overrides must be a mapping')
        experiment_base = base
        if 'base_yaml' in experiment:
            base_path = experiment['base_yaml']
            if not isinstance(base_path, str) or not base_path.strip():
                raise ValueError(f'{name}: base_yaml must be a nonempty path string')
            base_path = Path(base_path).expanduser()
            if not base_path.is_absolute():
                base_path = ROOT / base_path
            experiment_base = yaml.safe_load(base_path.read_text())
        if not isinstance(experiment_base, dict) or not isinstance(experiment_base.get('exp_setting'), dict):
            raise ValueError(f'{name}: base YAML must contain an exp_setting mapping')
        config = copy.deepcopy(experiment_base)
        config['exp_setting']['exp_name'] = name
        seen = {'exp_setting.exp_name'}
        for key, value in overrides.items():
            if not isinstance(key, str):
                raise ValueError('Override keys must be strings')
            if '.' not in key:
                matches = [section for section, values in experiment_base.items()
                           if isinstance(values, dict) and key in values]
                if len(matches) != 1:
                    raise ValueError(f'Unknown or ambiguous key: {key}')
                key = f'{matches[0]}.{key}'
            if key in seen:
                raise ValueError(f'Duplicate override (use name for exp_name): {key}')
            seen.add(key)
            node = config
            path = key.split('.')
            for part in path[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f'Unknown config path: {key}')
                node = node[part]
            if not isinstance(node, dict) or path[-1] not in node:
                raise ValueError(f'Unknown config key: {key}')
            node[path[-1]] = copy.deepcopy(value)
        configs.append(config)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=ROOT/'configs/stage1_visual_encoder_visloc.yaml')
    parser.add_argument('--batch-config', '--cfg-var-file', dest='batch_config', type=Path,
                        default=HERE/'stage1_batch_config.yaml')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base = yaml.safe_load(args.base_yaml.read_text())
    configs = build_configs(base, args.batch_config.read_text())
    if not configs:
        print('No enabled experiments; nothing to run.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output_dir or ROOT/'gen_fm_exps/logs/stage1_batch'/stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    for i, config in enumerate(configs, 1):
        path = out/f'{i:03d}.yaml'
        # Never silently overwrite a previously prepared experiment.
        with path.open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        command = [sys.executable, str(HERE/'run.py'), '--p_yaml', str(path)]
        commands.append(command)
        print(f'[{i}/{len(configs)}] {config["exp_setting"]["exp_name"]}\n{shlex.join(command)}', flush=True)
    if not args.dry_run:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
