"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 22:34
# @Author  : Chen Wang
# @Site    : 
# @File    : main_trainer.py
# @Email   : chenwang99@buaa.edu.cn
# @details : 
*****************************************************
"""
from utils.args import Args
from utils.tools import set_seed


class Trainers:
    def __init__(self, args_oj=None):
        self.args_oj = args_oj if args_oj is not None else Args()
        self.args = self.args_oj.get_args()
        self.base_seed = self._set_base_seed()
        self.instance_trainer = self._create_trainer()

    def get_trainer(self):
        return self.instance_trainer


    def test(self):
        return self.instance_trainer.run_online_inference_only()

    def get_pretrained_enhanced_trainer_model(self):
        model = self.instance_trainer.get_pretrained_enhanced_model()
        return model, self.instance_trainer

    def train_test(self):
        if self.args.online_inference_only:
            return self.test()
        if self.args.base_model == 'MegaCRN':
            return self.instance_trainer.traintest_model()
        else:
            raise ValueError('Please enter the valid base model name!')

    def _set_base_seed(self):
        """
        Prefer the seed reported by the base model.
        Fall back to the STBoost initialization seed when no base-model seed is provided.
        """
        base_seed = getattr(self.args, 'base_seed', None)
        if base_seed is not None:
            set_seed(base_seed)
            return base_seed

        set_seed(self.args.stb_init_seed)
        return self.args.stb_init_seed

    def _create_trainer(self):
        if self.args.base_model == 'MegaCRN':
            return self._init_MegaCRN_trainer()
        else:
            raise ValueError('Please enter the valid base model name!')

    def _init_MegaCRN_trainer(self):
        from trainers.MegaCRN_trainer import MegaCRN_Trainer
        return MegaCRN_Trainer(self.args, self.base_seed)
