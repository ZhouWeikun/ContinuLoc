import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage3_yaml",
        default=os.path.join(PROJECT_ROOT, "configs/pose_inference.yaml"),
    )
    parser.add_argument("--stage2_opts_yaml", default=None)
    parser.add_argument("--use_train_uav", action="store_true")
    args, remaining_argv = parser.parse_known_args(argv)

    from continuloc.engine.compat_runtime import preload_conda_libstdcpp

    preload_conda_libstdcpp()
    from continuloc.engine.pose_inference import MinimalStage3INGPInferencer, _find_latest_epoch_ckpt

    if "--p_yaml" not in remaining_argv:
        remaining_argv = ["--p_yaml", args.stage3_yaml] + remaining_argv
    else:
        raise ValueError("Use --stage3_yaml for Stage3 minimal config; --p_yaml is kept internal.")

    import yaml
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--inherit_stage2_yaml')
    selector.add_argument('--load_stage2_ckpt')
    selector.add_argument('--selected_scene_name')
    selector.add_argument('--selected_scene_names', nargs='+')
    selector.add_argument('--p_grid_config_yaml')
    selection, _ = selector.parse_known_args(remaining_argv)
    stage3_path = Path(args.stage3_yaml).expanduser()
    if not stage3_path.is_absolute():
        stage3_path = Path(PROJECT_ROOT)/stage3_path
    stage3 = yaml.safe_load(stage3_path.read_text())
    exp = stage3.get('exp_setting', {})
    stage2_yaml = selection.inherit_stage2_yaml or args.stage2_opts_yaml or exp.get('inherit_stage2_yaml')
    if stage2_yaml:
        stage2_path = Path(stage2_yaml).expanduser()
        if not stage2_path.is_absolute():
            stage2_path = Path(PROJECT_ROOT)/stage2_path
        if not selection.inherit_stage2_yaml:
            remaining_argv.extend(['--inherit_stage2_yaml', str(stage2_path)])
        if not (selection.load_stage2_ckpt or exp.get('load_stage2_ckpt')):
            maybe_ckpt = _find_latest_epoch_ckpt(str(stage2_path.parent))
            if maybe_ckpt:
                remaining_argv.extend(["--load_stage2_ckpt", maybe_ckpt])
        # The shared Stage 3 YAML owns current manifest paths. Infer the single
        # scene/model layout from the checkpoint without expanding its old paths.
        from continuloc.config.inference import checkpoint_scene_name, checkpoint_grid_path
        stage2 = yaml.safe_load(stage2_path.read_text())
        settings = stage3.get('scenes_setting') or {}
        configured_name = stage3.get('exp_setting', {}).get('selected_scene_name') or settings.get('selected_scene_name')
        expected_scene = checkpoint_scene_name(stage2)
        selected = selection.selected_scene_name or configured_name
        multiple = selection.selected_scene_names or settings.get('selected_scene_names')
        if selection.selected_scene_names:
            selected = None  # explicit CLI list takes precedence over YAML
        if selected or multiple:
            chosen = checkpoint_scene_name({'scenes_setting': (
                {'selected_scene_name': selected} if selected else {'selected_scene_names': multiple})})
            if chosen != expected_scene:
                raise ValueError(f'Scene {chosen!r} differs from Stage 2 scene {expected_scene!r}')
        else:
            remaining_argv.extend(['--selected_scene_name', expected_scene])
        explicit_grid = selection.p_grid_config_yaml or stage3.get('network_setting', {}).get('p_grid_config_yaml')
        if not explicit_grid:
            remaining_argv.extend(['--p_grid_config_yaml', str(checkpoint_grid_path(stage2))])

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining_argv
        inferencer = MinimalStage3INGPInferencer()
        inferencer.test(use_train_uav=args.use_train_uav)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
