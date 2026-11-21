import os
import torch
import yaml
from collections.abc import Sequence

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE2_INHERIT_SCOPES = ('network', 'data', 'scenes', 'hardware')
STAGE2_INHERIT_SECTION_MAP = {
    'network': 'network_setting',
    'data': 'data_setting',
    'hardware': 'hardware_setting',
}

def _normalize_inherit_stage2_scope(scope_value):
        if not scope_value:
            scopes = ['network']
        elif isinstance(scope_value, str):
            scopes = [part.strip() for part in scope_value.split(',') if part.strip()]
        elif isinstance(scope_value, Sequence):
            scopes = []
            for item in scope_value:
                if isinstance(item, str):
                    scopes.extend(part.strip() for part in item.split(',') if part.strip())
                else:
                    scopes.append(str(item).strip())
        else:
            scopes = [str(scope_value).strip()]

        invalid = [scope for scope in scopes if scope not in STAGE2_INHERIT_SCOPES]
        if invalid:
            raise ValueError(
                f"Unsupported inherit_stage2_scope: {invalid}. "
                f"Supported scopes: {STAGE2_INHERIT_SCOPES}"
            )

        # Remove duplicates while preserving order.
        deduped = []
        for scope in scopes:
            if scope not in deduped:
                deduped.append(scope)
        return tuple(deduped or ('network',))

def _apply_inherit_stage2_yaml(opt):
        """Inherit scoped settings from Stage 2 YAML/opts before Stage 3 network initialization.

        Only network, data, scenes, and hardware settings are inherited, keeping
        Stage 3 experiment and learning settings independent. Inherited values
        fill only keys not explicitly declared in the local Stage 3 YAML."""
        inherit_yaml = getattr(opt, 'inherit_stage2_yaml', '')
        if not inherit_yaml:
            return opt

        inherit_yaml = os.path.abspath(inherit_yaml)
        if not os.path.exists(inherit_yaml):
            raise FileNotFoundError(f"inherit_stage2_yaml not found: {inherit_yaml}")
        inherit_scopes = _normalize_inherit_stage2_scope(
            getattr(opt, 'inherit_stage2_scope', 'network')
        )

        with open(inherit_yaml, 'r', encoding='utf-8') as f:
            stage2_cfg = yaml.safe_load(f) or {}

        explicit_stage3_keys = _collect_stage3_explicit_keys(
            getattr(opt, 'p_yaml', '')
        )
        explicit_stage3_exp_keys = explicit_stage3_keys.get('exp_setting', set())

        if bool(getattr(opt, 'inherit_stage1_fm_stage2', False)):
            stage2_exp_setting = stage2_cfg.get('exp_setting', {})
            if not isinstance(stage2_exp_setting, dict):
                stage2_exp_setting = {}

            stage1_fm_stage2_inherited = []
            stage1_fm_stage2_skipped = []

            if 'load_stage1_ckpt' in explicit_stage3_exp_keys:
                stage1_fm_stage2_skipped.append('load_stage1_ckpt')
            else:
                inherited_load_stage1_ckpt = stage2_exp_setting.get('load_stage1_ckpt', '')
                if inherited_load_stage1_ckpt:
                    opt.load_stage1_ckpt = inherited_load_stage1_ckpt
                    stage1_fm_stage2_inherited.append('load_stage1_ckpt')

            if 'inherit_stage1_yaml' in explicit_stage3_exp_keys:
                stage1_fm_stage2_skipped.append('inherit_stage1_yaml')
            else:
                inherited_stage1_yaml = stage2_exp_setting.get('inherit_stage1_yaml', '')
                if inherited_stage1_yaml:
                    opt.inherit_stage1_yaml = inherited_stage1_yaml
                    stage1_fm_stage2_inherited.append('inherit_stage1_yaml')

            if 'inherit_stage1_scope' in explicit_stage3_exp_keys:
                stage1_fm_stage2_skipped.append('inherit_stage1_scope')
            else:
                stage1_scope_override = str(getattr(opt, 'inherit_stage1_fm_stage2_scope', '') or '').strip()
                inherited_stage1_scope = stage1_scope_override or str(stage2_exp_setting.get('inherit_stage1_scope', '') or '').strip()
                if inherited_stage1_scope:
                    opt.inherit_stage1_scope = inherited_stage1_scope
                    stage1_fm_stage2_inherited.append('inherit_stage1_scope')

            if stage1_fm_stage2_inherited:
                print("✅ Stage 3经由Stage 2继承Stage 1默认参数")
                print(f"   keys: {', '.join(stage1_fm_stage2_inherited)}")
            if stage1_fm_stage2_skipped:
                print(f"   skip override by stage3 yaml | exp_setting: {', '.join(stage1_fm_stage2_skipped)}")

        inherited_summary = {}
        skipped_summary = {}
        for scope in inherit_scopes:
            if scope == 'scenes':
                if 'scenes_setting' in explicit_stage3_keys:
                    skipped_summary['scenes_setting'] = ['<explicit_in_stage3_yaml>']
                    continue
                scenes_cfg = stage2_cfg.get('scenes_setting')
                if scenes_cfg:
                    setattr(opt, 'scenes_setting', scenes_cfg)
                    inherited_summary['scenes_setting'] = list(scenes_cfg.keys())
                continue

            section_name = STAGE2_INHERIT_SECTION_MAP[scope]
            section_cfg = stage2_cfg.get(section_name, {})
            if not isinstance(section_cfg, dict):
                continue
            explicit_keys = explicit_stage3_keys.get(section_name, set())
            inherited_keys = []
            skipped_keys = []
            for key, value in section_cfg.items():
                if key in explicit_keys:
                    skipped_keys.append(key)
                    continue
                setattr(opt, key, value)
                inherited_keys.append(key)
            if inherited_keys:
                inherited_summary[section_name] = inherited_keys
            if skipped_keys:
                skipped_summary[section_name] = skipped_keys

        if inherited_summary:
            print(f"✅ 从Stage 2配置继承参数: {inherit_yaml}")
            print(f"   scopes: {', '.join(inherit_scopes)}")
            for section_name, keys in inherited_summary.items():
                print(f"   {section_name}: {', '.join(keys)}")
        else:
            print(f"⚠️  inherit_stage2_yaml未提供可继承的scope字段: {inherit_yaml}")
        for section_name, keys in skipped_summary.items():
            print(f"   skip override by stage3 yaml | {section_name}: {', '.join(keys)}")

        opt.inherit_stage2_yaml = inherit_yaml
        opt.inherit_stage2_scope = ','.join(inherit_scopes)
        return opt

def _load_yaml_dict(yaml_path):
        if not yaml_path:
            return {}
        yaml_path = str(yaml_path).strip()
        if not yaml_path:
            return {}
        yaml_path_abs = yaml_path if os.path.isabs(yaml_path) else os.path.join(project_root, yaml_path)
        if not os.path.exists(yaml_path_abs):
            return {}
        with open(yaml_path_abs, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}

def _merge_section_key_sets(target, source):
        for section_name, keys in source.items():
            target.setdefault(section_name, set()).update(keys)
        return target

def _collect_declared_keys_from_cfg(cfg):
        declared = {}
        exp_cfg = cfg.get('exp_setting', None)
        if isinstance(exp_cfg, dict):
            declared['exp_setting'] = set(exp_cfg.keys())
        for section_name in ('data_setting', 'network_setting', 'hardware_setting'):
            section_cfg = cfg.get(section_name, None)
            if isinstance(section_cfg, dict):
                declared[section_name] = set(section_cfg.keys())
        if isinstance(cfg.get('scenes_setting', None), dict):
            declared['scenes_setting'] = {'__section__'}
        return declared

def _collect_stage3_explicit_keys(yaml_path, _visited=None):
        yaml_path = str(yaml_path or '').strip()
        if not yaml_path:
            return {}
        yaml_path_abs = yaml_path if os.path.isabs(yaml_path) else os.path.join(project_root, yaml_path)
        if _visited is None:
            _visited = set()
        if yaml_path_abs in _visited:
            return {}
        _visited.add(yaml_path_abs)

        cfg = _load_yaml_dict(yaml_path_abs)
        if not cfg:
            return {}

        declared = {}
        base_yaml = cfg.get('p_yaml') or cfg.get('exp_setting', {}).get('p_yaml')
        if base_yaml:
            declared = _collect_stage3_explicit_keys(base_yaml, _visited=_visited)
        return _merge_section_key_sets(declared, _collect_declared_keys_from_cfg(cfg))

def _load_checkpoints_for_test(self):
        """Load checkpoints for evaluation.

        Load Stage 3 metric_net, Stage 2 grid/grid_mlp, and the pretrained
        Stage 1 vis_encoder/vis_aggregator."""
        import yaml

        print("\n" + "=" * 80)
        print("加载测试用的checkpoint")
        print("=" * 80)

        # --- 1. Load the current Stage 3 checkpoint ---
        stage3_ckpt_path = self._get_stage3_checkpoint_path()

        if stage3_ckpt_path:
            print(f"\n📦 Stage 3 checkpoint: {stage3_ckpt_path}")
            self._load_checkpoint(
                stage3_ckpt_path,
                self.param2optimize,
                mode='test'
            )
            self._load_loss_fn_temperature_from_ckpt(stage3_ckpt_path)
        else:
            raise ValueError("未找到Stage 3的checkpoint，无法进行测试。")

        # --- 2. Load the pretrained Stage 2 checkpoint ---
        stage2_ckpt_path = self._get_stage2_checkpoint_path(stage3_ckpt_path)

        if stage2_ckpt_path:
            print(f"\n📦 Stage 2 checkpoint: {stage2_ckpt_path}")
            self._load_checkpoint(
                stage2_ckpt_path,
                {'grid': self.grid, 'grid_mlp': self.grid_mlp},
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 2的checkpoint，无法进行测试。")

        # --- 3. Load the pretrained Stage 1 checkpoint ---
        stage1_ckpt_path = self._get_stage1_checkpoint_path(stage2_ckpt_path)

        if stage1_ckpt_path:
            print(f"\n📦 Stage 1 checkpoint: {stage1_ckpt_path}")
            self._load_checkpoint(
                stage1_ckpt_path,
                {'vis_encoder': self.vis_encoder, 'vis_aggregator': self.vis_aggregator},
                mode='test'
            )
        else:
            raise ValueError("未找到Stage 1的checkpoint，无法进行测试。")

        print("\n" + "=" * 80)
        print("✅ 所有checkpoint加载完成")
        print("=" * 80 + "\n")

def _load_loss_fn_temperature_from_ckpt(self, ckpt_path):
        """Read loss_fn beta from the checkpoint as the evaluation temperature."""
        if not ckpt_path:
            return

        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'loss_fn' not in ckpt:
            print("⚠️  checkpoint中未找到loss_fn，保持默认temperature")
            return

        loss_state = ckpt['loss_fn']
        self.loss_fn_state = loss_state

        beta = None
        if isinstance(loss_state, dict):
            if 'log_beta' in loss_state:
                beta = torch.exp(loss_state['log_beta'])
            elif 'fixed_beta' in loss_state:
                beta = loss_state['fixed_beta']

        if beta is None:
            print("⚠️  loss_fn中未找到beta，保持默认temperature")
            return

        beta_val = float(beta.detach().cpu().item())
        self.loss_fn_beta = beta_val
        self.energy_temperature = beta_val
        print(f"✅ 从loss_fn加载temperature(beta): {self.energy_temperature:.5f}")

def _load_stage2_checkpoint(self):
        """Load the grid trained in Stage 2."""
        print(f"\n加载Stage 2 checkpoint: {self.opt.load_stage2_ckpt}")

        self._load_checkpoint(
            self.opt.load_stage2_ckpt,
            {
                'grid': self.grid,
                'grid_mlp': self.grid_mlp
            }
        )

        print("✅ Stage 2模型加载完成\n")

def _get_stage3_checkpoint_path(self):
        """Resolve the Stage 3 checkpoint path."""
        import os

        # Prefer the explicitly supplied command-line path.
        if hasattr(self.opt, 'load2test') and self.opt.load2test:
            print(f"从opt.load2test读取: {self.opt.load2test}")
            return self.opt.load2test

        # Otherwise, find the latest checkpoint in the experiment directory.
        if self.exp_dir2save and os.path.exists(self.exp_dir2save):
            ckpts = [f for f in os.listdir(self.exp_dir2save) if f.startswith('epoch')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace('epoch', '').split('.')[0]))
                ckpt_path = os.path.join(self.exp_dir2save, ckpts[-1])
                print(f"从实验目录读取: {ckpt_path}")
                return ckpt_path

        print(f"⚠️  未找到Stage 3 checkpoint:")
        print(f"   opt.load2test = {getattr(self.opt, 'load2test', 'NOT SET')}")
        print(f"   exp_dir2save = {self.exp_dir2save}")
        return None

def _get_stage2_checkpoint_path(self, stage3_ckpt_path):
        """Resolve the Stage 2 checkpoint path.

        Prefer opt.load_stage2_ckpt, then the Stage 3 experiment's opts.yaml."""
        import yaml
        import os

        # Prefer the command-line argument.
        if hasattr(self.opt, 'load_stage2_ckpt') and self.opt.load_stage2_ckpt:
            return self.opt.load_stage2_ckpt

        # Read from the Stage 3 opts.yaml.
        if stage3_ckpt_path:
            stage3_exp_dir = os.path.dirname(stage3_ckpt_path)
            stage3_opts_path = os.path.join(stage3_exp_dir, 'opts.yaml')

            if os.path.exists(stage3_opts_path):
                try:
                    with open(stage3_opts_path, 'r') as f:
                        stage3_opts = yaml.safe_load(f)

                    if 'exp_setting' in stage3_opts:
                        stage2_path = stage3_opts['exp_setting'].get('load_stage2_ckpt')
                        if stage2_path:
                            print(f"从opts.yaml读取Stage 2路径: {stage3_opts_path}")
                            return stage2_path
                except Exception as e:
                    print(f"⚠️  读取opts.yaml失败: {e}")

        return None

def _get_stage1_checkpoint_path(self, stage2_ckpt_path):
        """Resolve the Stage 1 checkpoint path.

        Prefer opt.load_stage1_ckpt, then the Stage 2 experiment's opts.yaml."""
        import yaml
        import os

        # Prefer the command-line argument.
        if hasattr(self.opt, 'load_stage1_ckpt') and self.opt.load_stage1_ckpt:
            return self.opt.load_stage1_ckpt

        # Read from the Stage 2 opts.yaml.
        if stage2_ckpt_path:
            stage2_exp_dir = os.path.dirname(stage2_ckpt_path)
            stage2_opts_path = os.path.join(stage2_exp_dir, 'opts.yaml')

            if os.path.exists(stage2_opts_path):
                try:
                    with open(stage2_opts_path, 'r') as f:
                        stage2_opts = yaml.safe_load(f)

                    if 'exp_setting' in stage2_opts:
                        stage1_path = stage2_opts['exp_setting'].get('load_stage1_ckpt')
                        if stage1_path:
                            print(f"从opts.yaml读取Stage 1路径: {stage2_opts_path}")
                            return stage1_path
                except Exception as e:
                    print(f"⚠️  读取opts.yaml失败: {e}")

        return None
