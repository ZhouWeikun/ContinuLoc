import subprocess
import os
import yaml

def backup_experiment(exp_dir,opt=None):
    """Save the current Git commit hash and uncommitted changes to the experiment directory."""
    if not exp_dir or not os.path.isdir(exp_dir):
        print(f"跳过实验备份：目录不存在 {exp_dir}")
        return

    git_info_path = os.path.join(exp_dir, 'git_info.txt')

    print("正在保存 Git 信息...")

    try:
        # Get the commit hash.
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

        # Get the diff of uncommitted changes.
        # Use a || true to tolerate the status code from --quiet.
        diff = subprocess.check_output('git diff HEAD --stat', shell=True).strip().decode('utf-8')
        uncommitted_changes = subprocess.check_output('git diff HEAD', shell=True).strip().decode('utf-8')

        with open(git_info_path, 'w') as f:
            f.write(f"Commit Hash:\n{commit_hash}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Diff Summary:\n{diff}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Uncommitted Changes (Patch):\n\n{uncommitted_changes}\n")

        print(f"Git 信息已保存到 {git_info_path}")

    except Exception as e:
        print(f"无法获取 Git 信息: {e}")
        print("请确保项目是一个 Git 仓库，并且已安装 Git。")

    if opt is not None:  # save opts
        # Save all configuration groups and the complete scenes_setting.
        grouped_params = {}

        # Checkpoint parameter names used to filter empty values
        checkpoint_params = ['load2train', 'load2test', 'load_stage1_ckpt', 'load_stage2_ckpt']

        # 1. Save all parameters defined in group_dict.
        for group, params in opt.group_dict.items():
            grouped_params[group] = {}
            for param in params:
                if hasattr(opt, param):
                    value = getattr(opt, param)
                    # Save checkpoint parameters only when nonempty.
                    if param in checkpoint_params:
                        if value and value != "":  # Include only checkpoint parameters with values.
                            grouped_params[group][param] = value
                    else:
                        # Save all non-checkpoint parameters.
                        grouped_params[group][param] = value

        # 2. Add scenes_setting when present.
        # scenes_setting includes the full scene configuration and sampling strategy.
        if hasattr(opt, 'scenes_setting'):
            grouped_params['scenes_setting'] = opt.scenes_setting

        with open(f'{exp_dir}/opts.yaml', 'w') as fp:
            yaml.dump(grouped_params, fp, default_flow_style=False)

        print(f"配置已保存到 {exp_dir}/opts.yaml")
