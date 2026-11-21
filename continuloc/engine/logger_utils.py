#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Logging and experiment directory utilities.

Core utilities extracted from tool/utils_fm_duav.py."""

import os
import sys
import logging
from pathlib import Path


def get_unique_exp_dir(base_dir: str, exp_name: str) -> str:
    """Find an unused experiment directory name under base_dir.

    If base_dir/exp_name exists, try exp_name_1, exp_name_2, and so on.

    Args:
        base_dir: Root directory for experiments.
        exp_name: Desired experiment directory name.

    Returns:
        A unique directory name that does not yet exist."""
    # Convert the path string to a Path object.
    base_path = Path(base_dir)

    # 1. Check the original name first.
    original_path = base_path / exp_name
    if not original_path.exists():
        # Return the original path if it does not exist.
        return exp_name

    # 2. Otherwise, try numbered suffixes.
    counter = 1
    while True:
        # Construct a suffixed name, such as my_experiment_1.
        suffixed_name = f"{exp_name}_{counter}"
        new_path = base_path / suffixed_name

        # Check whether the suffixed path exists.
        if not new_path.exists():
            # Return the first unused path.
            return suffixed_name

        # Increment the suffix and retry if the path already exists.
        counter += 1


def get_logger(log_file='app.log', name='my_app'):
    """Configure an application logger isolated from third-party loggers.

    Args:
        log_file: Log file path.
        name: Logger name.

    Returns:
        The configured logger."""
    # Define log levels and formatting.
    log_level = logging.INFO
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    # 1. Obtain a dedicated named logger.
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 2. Disable propagation.
    # This prevents messages from reaching a root logger modified by libraries such as faiss.
    # Keep application logging independent.
    logger.propagate = False

    # 3. Add handlers only if they are not already present.
    if not logger.handlers:
        # Add the file handler.
        fh = logging.FileHandler(log_file, "a")  # Open the log in append mode ('a').
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Add the console stream handler.
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
