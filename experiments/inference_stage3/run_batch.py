"""Run independent Stage 3 YAML experiments sequentially; stop on failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

from batch_config import build_configs
from run import HERE, ROOT, execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--batch-config', '--cfg-var-file', type=Path, default=HERE / 'stage3_batch_config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    args.base_yaml = args.base_yaml.resolve()
    args.batch_config = args.batch_config.resolve()
    configs = build_configs(yaml.safe_load(args.base_yaml.read_text()), args.batch_config.read_text())
    if not configs:
        print('No enabled experiments; fill checkpoint paths and enable an experiment.')
        return
    # Validate every enabled group before starting any GPU work.
    for config in configs:
        prepare(config)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = ROOT / 'gen_fm_exps/logs/stage3_batch' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'batch.yaml').write_text(args.batch_config.read_text())
    summary = []
    try:
        for config in configs:
            try:
                summary.append(execute(config, dry_run=args.dry_run, python=args.python))
            except BaseException as error:
                summary.append({'name': config['exp_setting']['exp_name'], 'status': 'failed', 'error': str(error)})
                raise
    finally:
        path = directory / 'summary.json'
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'Batch summary: {path}')


if __name__ == '__main__':
    main()
