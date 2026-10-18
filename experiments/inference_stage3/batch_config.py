"""Strict independent configuration expansion for Stage 3 experiments."""
import copy
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]

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
        if not isinstance(name, str) or not name.strip() or name in names or '/' in name or '\\' in name or name in ('.', '..'):
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
