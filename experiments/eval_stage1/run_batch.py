"""Run sequential checkpoint evaluations from a YAML experiment list."""
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
        if not isinstance(experiment, dict) or set(experiment) - {'name', 'enabled', 'overrides'}:
            raise ValueError('Each experiment accepts name, enabled, overrides only')
        name = experiment.get('name')
        if (not isinstance(name, str) or not name.strip() or name in names
                or any(c in name for c in '/\\') or name in ('.', '..')):
            raise ValueError(f'Invalid or duplicate experiment name: {name!r}')
        names.add(name)
        enabled = experiment.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f'{name}: enabled must be a YAML boolean')
        if not enabled:
            continue
        overrides = experiment.get('overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError(f'{name}: overrides must be a mapping')
        config = copy.deepcopy(base)
        config['name'] = name
        seen = {'name'}
        for key, value in overrides.items():
            if not isinstance(key, str):
                raise ValueError('Override keys must be strings')
            if '.' not in key and key not in base:
                key = f'evaluation.{key}'
            if key in seen:
                raise ValueError(f'Duplicate override (set name outside overrides): {key}')
            seen.add(key)
            node = config
            path = key.split('.')
            for part in path[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f'Unknown config path: {key}')
                node = node[part]
            if not isinstance(node, dict) or path[-1] not in node:
                raise ValueError(f'Unknown config key: {key}')
            if isinstance(node[path[-1]], dict):
                raise ValueError(f'Override individual settings using dotted keys: {key}')
            node[path[-1]] = copy.deepcopy(value)
        configs.append(config)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--batch-config', '--cfg-var-file', dest='batch_config',
                        type=Path, default=HERE / 'eval_batch_config.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    configs = build_configs(yaml.safe_load(args.base_yaml.read_text()), args.batch_config.read_text())
    if not configs:
        print('No enabled experiments; nothing to run.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output_dir or ROOT / 'logs/eval_stage1_batch' / stamp).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    for i, config in enumerate(configs, 1):
        path = out / f'{i:03d}.yaml'
        with path.open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        command = [sys.executable, '-u', str(HERE / 'run.py'), '--config', str(path)]
        commands.append(command)
        print(f'[{i}/{len(configs)}] {config["name"]}\n{shlex.join(command)}', flush=True)
    if not args.dry_run:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
