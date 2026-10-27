"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2026-10-24 01:52:19
# @Author  : Chen Wang
# @Site    :
# @File    : checkpoint_config.py
# @Email   : chenwang99@buaa.edu.cn
# @details : Read portable checkpoint configurations for the normal main.py test path.
*****************************************************
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def load_checkpoint_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--test_only', '--online_inference_only', dest='test_only', type=int, default=0)
    parser.add_argument('--best_model_path')
    parser.add_argument('--checkpoint_config')
    options, _ = parser.parse_known_args(argv)
    if not options.test_only:
        if options.checkpoint_config:
            raise ValueError('--checkpoint_config requires --test_only 1')
        return list(argv)
    if not options.best_model_path:
        return list(argv)

    checkpoint = Path(options.best_model_path).resolve()
    config_path = (Path(options.checkpoint_config).resolve() if options.checkpoint_config
                   else checkpoint.with_name('config.json'))
    if not options.checkpoint_config and not config_path.exists():
        return list(argv)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Expected a version-1 checkpoint configuration with an argv string list')
    configured_argv = config.get('argv')
    if (config.get('version') != 1 or not isinstance(configured_argv, list)
            or not configured_argv or not all(isinstance(value, str) for value in configured_argv)):
        raise ValueError('Expected a version-1 checkpoint configuration with an argv string list')
    expected_hash = config.get('checkpoint_sha256')
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_hash):
            raise ValueError('Invalid checkpoint_sha256 in configuration')
        digest = hashlib.sha256()
        with checkpoint.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError('Checkpoint SHA256 does not match its configuration')

    # Saved training flags must not turn a checkpoint-only request into training.
    # Explicit command-line settings still take precedence over saved settings.
    return configured_argv + ['--test_only', '1', '--profile_online_inference', '1'] + list(argv)


def validate_checkpoint_inputs(args):
    if not args.online_inference_only or not args.profile_online_inference:
        raise ValueError('Checkpoint validation requires --test_only 1 and --profile_online_inference 1')
    if not args.best_model_path or not Path(args.best_model_path).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.best_model_path}')
    if args.start_date_str is None or args.end_date_str is None:
        raise ValueError('Checkpoint evaluation must specify dataset start/end dates')
    data_file = Path(args.traffic_df_filename)
    if not data_file.is_file():
        raise FileNotFoundError(f'Dataset HDF not found: {data_file.resolve()}')
