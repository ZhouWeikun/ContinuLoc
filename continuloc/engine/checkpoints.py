import os
import torch

def _resolve_save_dir(save_dir_or_exp_name):
    save_dir_or_exp_name = str(save_dir_or_exp_name)
    if (
        os.path.isabs(save_dir_or_exp_name)
        or os.path.dirname(save_dir_or_exp_name)
        or save_dir_or_exp_name.startswith(".")
    ):
        return save_dir_or_exp_name
    return os.path.join('./gen_fm_exps/ckpts', save_dir_or_exp_name)


def save_param(save_dir_or_exp_name, dict2save):
    """Save a parameter dictionary to a file.

    Convert PyTorch modules to their state_dict() representation automatically."""
    save_dir = _resolve_save_dir(save_dir_or_exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Prepare the state dictionary to save.
    state_dict_to_save = {}
    for key, value in dict2save.items():
        # Check for a module, optimizer, or other object exposing state_dict.
        if hasattr(value, 'state_dict'):
            # Serialize objects through state_dict().
            state_dict_to_save[key] = value.state_dict()
        else:
            # Keep metadata such as epoch and learning rate unchanged.
            state_dict_to_save[key] = value

    # 2. Resolve the filename and path using the existing logic.
    if 'epoch' not in state_dict_to_save:
        print("警告: 字典中缺少 'epoch' 信息，将使用 'latest'作为文件名。")
        epoch_label = 'latest'
    else:
        epoch_label = state_dict_to_save['epoch']

    if isinstance(epoch_label, int):
        save_filename = 'epoch%03d.pth' % epoch_label
    else:
        save_filename = 'epoch%s.pth' % epoch_label

    save_path = os.path.join(save_dir, save_filename)

    # 3. Save the processed state dictionary.
    torch.save(state_dict_to_save, save_path)
    print(f"参数已成功保存到: {save_path}")




def _adapt_state_dict_for_module(module, state_dict, module_name=""):
    """
    Adapt known backward-compatible checkpoint layouts before strict loading.

    Stage2GridFeatureMLP used to save the implementation module directly
    (e.g. input_encoder.weight). Newer code wraps it under ``impl`` and thus
    expects keys like ``impl.input_encoder.weight``.
    """
    if not hasattr(module, "state_dict") or not isinstance(state_dict, dict):
        return state_dict

    target_keys = set(module.state_dict().keys())
    source_keys = set(state_dict.keys())
    if not target_keys or not source_keys:
        return state_dict

    has_impl_targets = any(key.startswith("impl.") for key in target_keys)
    has_impl_source = any(key.startswith("impl.") for key in source_keys)
    if not has_impl_targets or has_impl_source:
        return state_dict

    converted = {}
    converted_count = 0
    for key, value in state_dict.items():
        impl_key = f"impl.{key}"
        if impl_key in target_keys:
            converted[impl_key] = value
            converted_count += 1
        else:
            converted[key] = value

    if converted_count > 0:
        label = f" for {module_name}" if module_name else ""
        print(f"Adapted legacy state_dict{label}: added 'impl.' prefix to {converted_count} keys")
        return converted

    return state_dict


def load_param(load_from, dict2load):
    print(f"Loading parameters from: {load_from}")
    checkpoint = torch.load(load_from, map_location=lambda storage, loc: storage) # Ensure device compatibility.

    for k, v in dict2load.items():
        if k in checkpoint:
            if k == 'epoch':
                dict2load['epoch'] = checkpoint['epoch']
                print(f"Loaded epoch: {dict2load['epoch']}")
            else:
                # Load parameters in place; assigning dict2load[k] = v is unnecessary.
                if hasattr(v, 'load_state_dict'): # Ensure the object implements this method.
                    state_dict = _adapt_state_dict_for_module(v, checkpoint[k], module_name=k)
                    v.load_state_dict(state_dict)
                    print(f"Loaded parameters for: {k}")
                else:
                    # Assign ordinary values directly when v is not a model or optimizer.
                    dict2load[k] = checkpoint[k]
        else:
            print(f"Warning: Key '{k}' not found in checkpoint file.")

    return checkpoint
