"""Evaluate a stage1 checkpoint using its saved model/data configuration."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--_worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    import yaml

    config_path = args.config.resolve()
    os.chdir(ROOT)
    cfg = yaml.safe_load(config_path.read_text())
    checkpoint = Path(cfg['checkpoint']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args._worker_output is None:
        name = cfg['name']
        if not isinstance(name, str) or not name.strip() or any(c in name for c in '/\\') or name in ('.', '..'):
            raise ValueError('name must be a nonempty directory name without path separators')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_id = f"{checkpoint.parent.name}_{checkpoint.stem}_{name}_{stamp}"
        output = Path(cfg['output_root']).resolve() / run_id
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / 'config.yaml'
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--config', str(snapshot), '--_worker-output', str(output)]
        # Capture native stderr, progress bars and tracebacks as well as Python prints.
        with (output / 'eval.log').open('x') as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        if returncode:
            raise SystemExit(returncode)
        return

    import torch
    from continuloc.config.parser import get_parse
    from continuloc.engine.stage1_visual_encoder_minimal import MinimalStage1VisualEncoderTrainer

    sys.argv = [sys.argv[0], '--p_yaml', str(checkpoint.parent / 'opts.yaml')]
    opt = get_parse()
    opt.load2test = str(checkpoint)
    for key, value in cfg['evaluation'].items():
        setattr(opt, key, value)
    memory_budget = getattr(opt, 'eval_cuda_memory_budget_gib', None)
    if memory_budget is not None and torch.cuda.is_available():
        budget = float(memory_budget)
        if not math.isfinite(budget) or budget <= 1:
            raise ValueError('eval_cuda_memory_budget_gib must be finite and greater than 1')
        device = torch.device('cuda:' + opt.gpu_ids[0])
        total = torch.cuda.get_device_properties(device).total_memory
        # Reserve 1 GiB for CUDA context/library memory outside PyTorch's allocator.
        torch.cuda.set_per_process_memory_fraction(min((budget - 1) * 2**30 / total, 1.0), device)
        print(f'CUDA allocator budget: {budget - 1:g} GiB; external headroom: 1 GiB', flush=True)
    selected = cfg.get('scenes', [])
    scenes = [s for s in opt.scenes_setting['scenes'] if not selected or s['name'] in selected]
    if not scenes or (selected and set(selected) != {s['name'] for s in scenes}):
        raise ValueError('Unknown or empty scene selection')
    output = args._worker_output
    print(f'OUTPUT_DIR={output}', flush=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'resolved_options.yaml').write_text(yaml.safe_dump(vars(opt), sort_keys=False))
    (output / 'checkpoint.json').write_text(json.dumps({'path': str(checkpoint), 'size': checkpoint.stat().st_size, 'mtime_ns': checkpoint.stat().st_mtime_ns}, indent=2))
    if cfg.get('export_retrieval', False):
        opt.retrieval_export_dir = str(output / 'retrieval')
        opt.retrieval_export_opts_yaml = str(output / 'resolved_options.yaml')
    trainer = MinimalStage1VisualEncoderTrainer(opt)
    trainer._load_eval_ckpt_if_needed()
    results = {}
    # Load one scene at a time to bound satellite raster memory.
    for scene in scenes:
        opt.scenes_setting['scenes'] = [scene]
        results.update(trainer.test(init_datasets=True, load_ckpt=False, restore_train=False))
        (output / 'metrics.json').write_text(json.dumps(results, indent=2))
        for key in list(vars(trainer)):
            if key.startswith(('sat_dataset', 'uav_dataset')):
                delattr(trainer, key)
        gc.collect()
        torch.cuda.empty_cache()
    (output / 'completed.json').write_text(json.dumps({'scenes': list(results)}, indent=2))
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
