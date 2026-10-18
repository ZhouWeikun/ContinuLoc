import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from continuloc.engine.stage1_visual_encoder_minimal import MinimalStage1VisualEncoderTrainer

def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n"}


def main():
    if "--p_yaml" not in " ".join(sys.argv):
        sys.argv.extend(["--p_yaml", "configs/stage1_visual_encoder_visloc.yaml"])

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test_only", nargs="?", const=True, default=False, type=_parse_bool_arg)
    parser.add_argument("--exp_name_override", default="", type=str,help=" 如果命令行传入了exp_name_override 参数，则用该值替换 opt.exp_name；在不修改 yaml 配置文件的情况下，通过命令行为同一套配置启动不同命名的实验，方便对比实验的日志/checkpoint 目录区分")
    args, remaining_argv = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv
    from continuloc.config.parser import get_parse

    opt = get_parse()
    if args.exp_name_override:
        opt.exp_name = args.exp_name_override

    trainer = MinimalStage1VisualEncoderTrainer(opt=opt)
    if args.test_only:
        trainer.test(init_datasets=True, load_ckpt=True, restore_train=False)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
