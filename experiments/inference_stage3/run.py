"""Prepare and run one fixed coarse -> contraction -> CMA experiment."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from continuloc.config.parser import _expand_selected_scene_config


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve()


def archive_run_config(config, metadata, output, *, backfilled=False):
    """Keep the submitted configuration and its inheritance sources beside results."""
    output = Path(output)
    config_path = output / 'config.yaml'
    with config_path.open('x') as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    exp = config['exp_setting']
    stage2_path = project_path(exp['inherit_stage2_yaml'])
    stage2 = yaml.safe_load(stage2_path.read_text())
    sources = {'stage2_opts': str(stage2_path),
               'recall_thresholds': str(project_path(exp['stage3_recall_cfg_yaml']))}
    stage1_opts = exp.get('inherit_stage1_yaml')
    if 'inherit_stage1_yaml' not in exp and exp.get('inherit_stage1_fm_stage2', False):
        stage1_opts = stage2.get('exp_setting', {}).get('inherit_stage1_yaml')
    if stage1_opts:
        sources['stage1_opts'] = str(project_path(stage1_opts))
    snapshots = {}
    for name, source in sources.items():
        path = Path(source)
        if path.is_file():
            target = output / f'{name}.yaml'
            with target.open('xb') as stream:
                stream.write(path.read_bytes())
            snapshots[name] = str(target)
    metadata.update(result_config=str(config_path), config_sources=sources,
                    config_snapshots=snapshots,
                    config_archive_note=('Backfilled from original run config/log manifest; source YAML snapshots reflect files at backfill time.'
                                         if backfilled else 'Source YAML snapshots captured before inference.'),
                    config_archived_at=datetime.now(timezone.utc).isoformat())
    return metadata


def prepare(config):
    os.chdir(ROOT)
    config = copy.deepcopy(config)
    exp = config['exp_setting']
    inference = config.get('inference_setting', {})
    if inference.get('hds_scale_select_scope') == 'global_pose_max':
        if inference.get('hds_stage1_gallery_feature_source') not in {'visual_encoder', 'ingp'}:
            raise ValueError(
                "hds_scale_select_scope='global_pose_max' requires "
                "hds_stage1_gallery_feature_source='visual_encoder' or 'ingp'."
            )
        if inference.get('hds_stage1_visual_score_mode') != 'cosine_softmax':
            raise ValueError(
                "hds_scale_select_scope='global_pose_max' requires "
                "hds_stage1_visual_score_mode='cosine_softmax'."
            )
    for key in ('load_stage2_ckpt', 'inherit_stage2_yaml'):
        if not exp.get(key):
            raise ValueError(f'Fill exp_setting.{key} before running (including dry-run)')
        path = project_path(exp[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        exp[key] = str(path)
    stage2 = yaml.safe_load(Path(exp['inherit_stage2_yaml']).read_text())
    scenes = _expand_selected_scene_config(stage2.get('scenes_setting'))
    if not scenes or len(scenes.get('scenes', [])) != 1:
        raise ValueError('Stage 3 requires a single-scene Stage 2 opts.yaml')
    scene = scenes['scenes'][0]['name']
    selected = str(exp.get('selected_scene_name') or scene)
    if selected.isdigit() and scene.startswith('visloc_'):
        selected = f'visloc_{int(selected):02d}'
    if selected != scene:
        raise ValueError(f'Scene {selected!r} differs from Stage 2 scene {scene!r}')
    if 'scenes_setting' in config and config['scenes_setting'] != scenes:
        raise ValueError('Scene configuration must be inherited from Stage 2 opts.yaml')
    config['scenes_setting'] = scenes
    exp['selected_scene_name'] = scene
    teacher = exp.get('load_stage1_ckpt') or stage2.get('exp_setting', {}).get('load_stage1_ckpt')
    if not teacher or not project_path(teacher).is_file():
        raise FileNotFoundError(f'Stage 1 checkpoint missing: {teacher!r}')
    exp['load_stage1_ckpt'] = str(project_path(teacher))
    threshold_path = project_path(exp['stage3_recall_cfg_yaml'])
    thresholds = yaml.safe_load(threshold_path.read_text())
    selector = exp.get('stage3_recall_cfg', 'per_scene')
    if selector.lower() in ('per_scene', 'scene', 'auto'):
        selector = thresholds.get('per_scene', {}).get(scene, thresholds.get('default', 'cfg01'))
    threshold = thresholds['configs'][selector]
    exp['stage3_recall_cfg_yaml'] = str(threshold_path)
    print(f'Scene: {scene}; recall: {selector} = {threshold}', flush=True)
    return config, {'scene': scene, 'recall_config': selector, 'thresholds': threshold}


def execute(config, *, dry_run=False, python=sys.executable):
    config, metadata = prepare(config)
    exp = config['exp_setting']
    name = exp['exp_name']
    if not isinstance(name, str) or not name.strip() or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('exp_name must be a nonempty directory name')
    run_id = name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = project_path(exp.get('stage3_analysis_export_root') or f'gen_fm_exps/inference_stage3/{run_id}')
    logs = ROOT / 'gen_fm_exps/logs/inference_stage3' / run_id
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    exp.update(exp_name=run_id, stage3_analysis_export_root=str(output), dir2save_log=str(logs))
    config_path = logs / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    command = [python, str(ROOT / 'scripts/localize_stage3.py'), '--stage3_yaml', str(config_path)]
    metadata.update(run_id=run_id, output_dir=str(output), config=str(config_path), command=command,
                    stage2_checkpoint=exp['load_stage2_ckpt'], stage2_opts=exp['inherit_stage2_yaml'],
                    stage1_checkpoint=exp['load_stage1_ckpt'], status='prepared')
    archive_run_config(config, metadata, output)
    manifest = logs / 'manifest.json'
    def save():
        text = json.dumps(metadata, indent=2, ensure_ascii=False)
        manifest.write_text(text)
        (output / 'manifest.json').write_text(text)
    save()
    print(f'Prepared {run_id}\nConfig: {config_path}\nOutput: {output}', flush=True)
    if dry_run:
        return metadata
    start = time.perf_counter()
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    executable = Path(shutil.which(python) or python).absolute()
    env_lib = executable.parent.parent / 'lib'
    if executable.parent.name == 'bin' and env_lib.is_dir():
        environment['LD_LIBRARY_PATH'] = str(env_lib) + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
    try:
        metadata['status'] = 'running'
        save()
        with (logs / 'console.log').open('w') as log, (output / 'console.log').open('x') as result_log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=environment) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    result_log.write(line)
                    result_log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
        metadata['recall'] = json.loads((output / 'stage3_progressive_recall_report.json').read_text())
        metadata['timing'] = json.loads((output / 'timing.json').read_text())
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='failed', error=str(error))
        raise
    finally:
        metadata['wall_time_s'] = time.perf_counter() - start
        save()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '--stage3_yaml', '--p_yaml', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    execute(yaml.safe_load(args.config.read_text()), dry_run=args.dry_run, python=args.python)


if __name__ == '__main__':
    main()
