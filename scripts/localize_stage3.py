import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from continuloc.engine.stage3_ingp_inferencer_minimal import MinimalStage3INGPInferencer, _find_latest_epoch_ckpt

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage3_yaml",
        default=os.path.join(PROJECT_ROOT, "configs/stage3_ingp_minimal.yaml"),
    )
    parser.add_argument("--stage2_opts_yaml", default=None)
    parser.add_argument("--use_train_uav", action="store_true")
    args, remaining_argv = parser.parse_known_args(argv)

    if "--p_yaml" not in remaining_argv:
        remaining_argv = ["--p_yaml", args.stage3_yaml] + remaining_argv
    else:
        raise ValueError("Use --stage3_yaml for Stage3 minimal config; --p_yaml is kept internal.")

    if args.stage2_opts_yaml:
        if "--inherit_stage2_yaml" not in remaining_argv:
            remaining_argv.extend(["--inherit_stage2_yaml", args.stage2_opts_yaml])
        if "--load_stage2_ckpt" not in remaining_argv:
            maybe_ckpt = _find_latest_epoch_ckpt(os.path.dirname(os.path.abspath(args.stage2_opts_yaml)))
            if maybe_ckpt:
                remaining_argv.extend(["--load_stage2_ckpt", maybe_ckpt])

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + remaining_argv
        inferencer = MinimalStage3INGPInferencer()
        inferencer.test(use_train_uav=args.use_train_uav)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
