import os
import shutil
import argparse
from typing import List, Optional, Set


def _list_subdirs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )


def _find_prefix_ckpt(log_name: str, ckpt_names: Set[str]) -> Optional[str]:
    """Return ckpt_name when log_name has the form '{ckpt_name}_{extra}'.

    Return None if no checkpoint-name prefix matches."""
    for ckpt_name in ckpt_names:
        if log_name.startswith(ckpt_name + "_"):
            return ckpt_name
    return None


def _collect_ckpt_exp_names(
    ckpt_root: str,
    require_checkpoint_file: bool = False,
) -> Set[str]:
    exp_names: Set[str] = set()
    for exp_name in _list_subdirs(ckpt_root):
        ckpt_dir = os.path.join(ckpt_root, exp_name)
        if not require_checkpoint_file:
            exp_names.add(exp_name)
            continue

        has_ckpt_file = any(
            filename.endswith(".pth")
            and os.path.isfile(os.path.join(ckpt_dir, filename))
            for filename in os.listdir(ckpt_dir)
        )
        if has_ckpt_file:
            exp_names.add(exp_name)
    return exp_names


def clean_logs_without_ckpt(
    ckpt_root: str,
    log_root: str,
    dry_run: bool = True,
    require_checkpoint_file: bool = False,
) -> List[str]:
    """Remove log directories without a corresponding checkpoint experiment.

    Args:
        ckpt_root: Checkpoint root directory.
        log_root: Log root directory.
        dry_run: Print actions without deleting files when True.
        require_checkpoint_file: Require at least one .pth file in the
            checkpoint directory for an experiment to count as present.
    Returns:
        Names of log subdirectories selected for deletion."""
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: '{ckpt_root}'")
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root not found: '{log_root}'")

    ckpt_exp_names = _collect_ckpt_exp_names(
        ckpt_root,
        require_checkpoint_file=require_checkpoint_file,
    )
    log_exp_names = _list_subdirs(log_root)

    # Exact match: retain a log directory whose name equals the checkpoint directory name.
    # Prefix with an extra suffix: delete log directories named ckpt_name_xx.
    # No match: delete.
    logs_to_delete = [
        exp_name for exp_name in log_exp_names
        if exp_name not in ckpt_exp_names
    ]

    # Separate deletion candidates with a matching prefix from entirely unrelated names.
    prefix_variant_logs = []   # Log name equals a checkpoint name plus '_xx'.
    unrelated_logs = []        # Log name has no matching checkpoint-name prefix.
    for exp_name in logs_to_delete:
        matched_ckpt = _find_prefix_ckpt(exp_name, ckpt_exp_names)
        if matched_ckpt is not None:
            prefix_variant_logs.append((exp_name, matched_ckpt))
        else:
            unrelated_logs.append(exp_name)

    print("-" * 60)
    print(f"Checkpoint root: {ckpt_root}")
    print(f"Log root: {log_root}")
    print(f"Found {len(ckpt_exp_names)} valid ckpt experiment folders.")
    print(f"Found {len(log_exp_names)} log folders.")
    print(f"Unmatched log folders: {len(logs_to_delete)} "
          f"(unrelated: {len(unrelated_logs)}, prefix-variant: {len(prefix_variant_logs)})")
    print("-" * 60)

    if not logs_to_delete:
        print("No useless log folders found.")
        return []

    if dry_run:
        print("DRY RUN MODE: No folders will be deleted.")
    else:
        print("Deleting unmatched log folders...")

    for exp_name in logs_to_delete:
        log_dir = os.path.join(log_root, exp_name)
        if dry_run:
            print(f"  - Would delete: {log_dir}")
            continue
        shutil.rmtree(log_dir)
        print(f"  - Deleted: {log_dir}")

    print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
    action_verb = "to be deleted" if dry_run else "deleted"
    if unrelated_logs:
        print(f"[No ckpt match] Log folders {action_verb} ({len(unrelated_logs)}):")
        for exp_name in unrelated_logs:
            print(f"  - {exp_name}")
    if prefix_variant_logs:
        print(f"[Prefix match, extra suffix] Log folders {action_verb} ({len(prefix_variant_logs)}):")
        for exp_name, base_ckpt in prefix_variant_logs:
            print(f"  - {exp_name}  (base ckpt: {base_ckpt})")
    print("=" * 49)

    return logs_to_delete


if __name__ == "__main__":
    # clean manually:
    ckpt_root = '/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts'
    log_dir = '/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/logs'
    clean_logs_without_ckpt(
        ckpt_root=ckpt_root,
        log_root=log_dir,
        dry_run=False,
        require_checkpoint_file=True,
    )
