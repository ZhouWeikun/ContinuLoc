"""Run named XMU evaluation configurations sequentially with isolated outputs."""
import argparse
import copy
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_configs(base, plan):
    if not isinstance(plan, dict) or set(plan) != {'experiments'} or not isinstance(plan['experiments'], list):
        raise ValueError('Batch config must contain only an experiments list')
    result, names = [], set()
    for entry in plan['experiments']:
        if not isinstance(entry, dict) or set(entry) - {'name', 'enabled', 'overrides'}:
            raise ValueError('Each entry accepts name, enabled, overrides')
        name = entry.get('name')
        if not isinstance(name, str) or not name.strip() or name in ('.', '..') or '/' in name or '\\' in name or name in names:
            raise ValueError(f'Invalid or duplicate name: {name!r}')
        names.add(name)
        enabled = entry.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be a boolean')
        overrides = entry.get('overrides', {})
        if not isinstance(overrides, dict) or set(overrides) - set(base):
            raise ValueError(f'{name}: unknown override keys or invalid mapping')
        if enabled:
            config = copy.deepcopy(base)
            config.update(overrides)
            result.append((name, config))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=HERE/'config.yaml')
    parser.add_argument('--batch-config', type=Path, default=HERE/'batch_config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-queries', type=int, help='Override every experiment for smoke checks')
    parser.add_argument('--stride-m', type=float, help='Override every experiment for smoke checks')
    args = parser.parse_args()
    configs = build_configs(yaml.safe_load(args.base_yaml.read_text()), yaml.safe_load(args.batch_config.read_text()))
    if not configs:
        print('No enabled experiments.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs = ROOT/'gen_fm_exps/logs/xmu_xy_batch'/stamp
    logs.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'prepared', 'experiments': []}
    jobs = []
    for name, config in configs:
        for key in ('max_queries', 'stride_m'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        output = Path(config['output_root'])
        if not output.is_absolute():
            output = ROOT/output
        config['output_root'] = str(output/'batch'/stamp/name)
        path = logs/f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        record = {'name': name, 'config': str(path), 'log': str(logs/f'{name}.log'),
                  'output_root': config['output_root'], 'status': 'pending'}
        manifest['experiments'].append(record)
        jobs.append((config, record))
    def save():
        (logs/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    print(f'BATCH_DIR={logs}', flush=True)
    if args.dry_run:
        print('Dry-run: configs generated; no inference.')
        return
    summary = []
    for config, record in jobs:
        manifest['status'] = 'running'
        record['status'] = 'running'
        save()
        print(f"Running {record['name']}", flush=True)
        with open(record['log'], 'x') as stream:
            process = subprocess.run([sys.executable, '-u', str(HERE/'evaluate_xy.py'),
                                      '--config', record['config']], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        completed = list(Path(record['output_root']).glob('*/completed.json'))
        if process.returncode or len(completed) != 1:
            record['status'] = manifest['status'] = 'failed'
            record['returncode'] = process.returncode
            save()
            raise SystemExit(f"Failed: {record['name']}; see {record['log']}")
        output = completed[0].parent
        metrics = json.loads((output/'metrics.json').read_text())
        record.update(status='complete', output_dir=str(output), metrics=metrics)
        row = {'name': record['name'], 'mode': config['rot_retrieval_mode'], 'output_dir': str(output),
               **{k:v for k,v in metrics.items() if k != 'recall_percent'}}
        for threshold, recalls in metrics['recall_percent'].items():
            for k, value in recalls.items():
                row[f'recall_{threshold}m_at_{k}'] = value
        summary.append(row)
        with (logs/'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in summary for k in row)))
            writer.writeheader(); writer.writerows(summary)
        save()
        print(f"Completed {record['name']}: R@1/50m={metrics['recall_percent']['50']['1']:.2f}%", flush=True)
    manifest['status'] = 'complete'
    save()
    print(f'Completed batch: {logs}', flush=True)

if __name__ == '__main__':
    main()
