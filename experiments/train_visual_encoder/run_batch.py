"""Run sequential Stage 1 experiments described by a YAML experiment list."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continuloc.config.batch import build_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaml', type=Path, default=ROOT/'configs/visual_encoder_continuscenes.yaml')
    parser.add_argument('--batch-config', '--cfg-var-file', dest='batch_config', type=Path,
                        default=HERE/'visual_encoder_batch.yaml')
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
        command = [sys.executable, '-m', 'continuloc.cli.train_visual_encoder', '--p_yaml', str(path)]
        commands.append(command)
        print(f'[{i}/{len(configs)}] {config["exp_setting"]["exp_name"]}\n{shlex.join(command)}', flush=True)
    if not args.dry_run:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
