import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n"}


def main(default_yaml="configs/neural_field_continuscenes.yaml"):
    if not any(arg == "--p_yaml" or arg.startswith("--p_yaml=") for arg in sys.argv[1:]):
        sys.argv.extend(["--p_yaml", default_yaml])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", nargs="?", const=True, default=False, type=_parse_bool_arg)
    parser.add_argument("--exp_name_override", default="", type=str)
    args, remaining_argv = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    from continuloc.config.parser import get_parse

    opt = get_parse()
    if args.exp_name_override:
        opt.exp_name = args.exp_name_override
    scenes = getattr(opt, "scenes_setting", {}).get("scenes", [])
    if len(scenes) != 1:
        raise ValueError("Stage 2 trains one scene per field; select exactly one scene.")
    required_paths = ["load_stage1_ckpt", "inherit_stage1_yaml"]
    if args.test_only:
        if not (opt.load2test or opt.load2train):
            raise ValueError("--test_only requires --load2test or --load2train.")
    for key in required_paths + ["load2train", "load2test"]:
        value = getattr(opt, key, "")
        if not value and key in required_paths:
            raise ValueError(f"Set --{key} to the Stage 1 {'opts.yaml' if key == 'inherit_stage1_yaml' else '.pth checkpoint'}.")
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = Path(PROJECT_ROOT) / path
            if not path.is_file():
                raise FileNotFoundError(f"{key}: {path}")
            if key != "inherit_stage1_yaml" and path.suffix in {".yaml", ".yml"}:
                raise ValueError(f"{key} requires model weights, not a YAML file: {path}")
            setattr(opt, key, str(path))

    from continuloc.engine.compat_runtime import preload_conda_libstdcpp

    preload_conda_libstdcpp()
    from continuloc.engine.neural_field_trainer import MinimalStage2INGPTrainer

    trainer = MinimalStage2INGPTrainer(opt=opt)
    if args.test_only:
        trainer.test(init_datasets=True, load_ckpt=True, restore_train=False)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
