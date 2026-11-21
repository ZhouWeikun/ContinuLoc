"""Train or evaluate the visual encoder using the shared configuration parser."""
import argparse
import sys

from continuloc.config.parser import _parse_boolean, get_parse


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_only', nargs='?', const=True, default=False, type=_parse_boolean)
    parser.add_argument('--exp_name_override', default='')
    args, remaining = parser.parse_known_args(argv)
    if not any(arg == '--p_yaml' or arg.startswith('--p_yaml=') for arg in remaining):
        remaining = ['--p_yaml', 'configs/visual_encoder_continuscenes.yaml', *remaining]
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        opt = get_parse()
        if args.exp_name_override:
            opt.exp_name = args.exp_name_override
        from continuloc.engine.compat_runtime import preload_conda_libstdcpp
        preload_conda_libstdcpp()
        from continuloc.engine.visual_encoder_trainer import MinimalStage1VisualEncoderTrainer
        trainer = MinimalStage1VisualEncoderTrainer(opt=opt)
        if args.test_only:
            trainer.test(init_datasets=True, load_ckpt=True, restore_train=False)
        else:
            trainer.train()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
