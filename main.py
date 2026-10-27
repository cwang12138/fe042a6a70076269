"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 13:37
# @Author  : Chen Wang
# @Site    : 
# @File    : main.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
import importlib

from utils.args import Args
from utils.checkpoint_config import validate_checkpoint_inputs


def main():
    args_oj = Args()
    args = args_oj.get_args()
    if args.check_only:
        validate_checkpoint_inputs(args)
        importlib.import_module(f'trainers.{args.base_model}_trainer')
        print(f'Checkpoint check passed: {args.base_model}, enhanced={args.is_enhanced}, '
              f'{args.dataset}, {args.p_len}-{args.q_len}. No training or inference started.')
        return
    from main_trainer import Trainers
    main_trainer = Trainers(args_oj=args_oj)
    return main_trainer.train_test()


if __name__ == '__main__':
    model = main()
