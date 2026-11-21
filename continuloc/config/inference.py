"""Resolve checkpoint identity and model structure independently of saved data paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_scene_name(stage2):
    settings = stage2.get('scenes_setting') or {}
    name = settings.get('selected_scene_name')
    if not name:
        selected = settings.get('selected_scene_names')
        if selected is not None:
            if len(selected) != 1:
                raise ValueError('Stage 3 requires a single-scene Stage 2 opts.yaml')
            name = selected[0]
        else:
            scenes = settings.get('scenes') or []
            if len(scenes) != 1:
                raise ValueError('Stage 3 requires a single-scene Stage 2 opts.yaml')
            name = scenes[0]['name']
    name = str(name).strip()
    if not name:
        raise ValueError('Stage 2 scene name is empty')
    return f'visloc_{int(name):02d}' if name.isdigit() else name


def checkpoint_grid_path(stage2, explicit=None):
    grid = explicit or stage2.get('network_setting', {}).get('p_grid_config_yaml')
    if not grid:
        raise ValueError('Set network_setting.p_grid_config_yaml or record it in Stage 2 opts')
    path = (ROOT/Path(grid).expanduser()).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Grid configuration not found: {path}. Set network_setting.p_grid_config_yaml "
            "to the grid configuration matching this checkpoint, or use its reproduction bundle."
        )
    return path
