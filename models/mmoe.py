#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :mmoe.py
@Description  :
@Date         :2021/10/15 14:27:36
@Author       :Arctic Little Pig
@Version      :1.0
'''

import numpy
import torch
import torch.nn as nn
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN
from deepctr_torch.layers.core import PredictionLayer

from ..base.mtlbasemodel import MTLBaseModel


class MMOE(MTLBaseModel):
    def __init__(self,
                 dnn_feature_columns, num_experts=3,
                 expert_dnn_hidden_units=(256, 128),
                 tower_dnn_hidden_units=(64, ),
                 gate_dnn_hidden_units=(),
                 l2_reg_embedding=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False,
                 task_types=('binary', 'binary'),
                 task_names=('ctr', 'ctcvr'),
                 device='cpu', gpus=None):

        super(MMOE, self).__init__(dnn_feature_columns, expert_dnn_hidden_units=expert_dnn_hidden_units,
                                   tower_dnn_hidden_units=tower_dnn_hidden_units, gate_dnn_hidden_units=gate_dnn_hidden_units,
                                   l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                   seed=seed, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation, dnn_use_bn=dnn_use_bn,
                                   task_types=task_types, task_names=task_names, device=device, gpus=gpus)

        self.num_experts = num_experts
        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")

        # build expert layer
        self.experts = nn.ModuleList([DNN(self.compute_input_dim(dnn_feature_columns), expert_dnn_hidden_units, activation=dnn_activation,
                                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for i in range(num_experts)])

        self.softmax = nn.Softmax(dim=1)
        self.mmoe_gates = nn.ParameterList([nn.Parameter(torch.randn(self.compute_input_dim(
            dnn_feature_columns), num_experts), requires_grad=True) for i in range(self.num_tasks)])

        self.towers = nn.ModuleList([DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                    dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device) for i in range(self.num_tasks)])
        self.linear = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1], 1, bias=False) for i in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task_type, )
                                 for task_type in task_types])

    def forword(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        experts_out = [expert(dnn_input) for expert in self.experts]
        expert_out = torch.stack(experts_out)

        mmoe_outs = [self.softmax(dnn_input @ gate)
                     for gate in self.mmoe_gates]

        task_outs = []
        for task_type, mmoe_out, tower, linear, out in zip(self.task_types, mmoe_outs, self.towers, self.linear, self.out):
            # build tower layer
            tower_input = mmoe_out.t().unsqueeze(
                2).expand(-1, -1, self.expert_dnn_hidden_units[-1]) * expert_out
            tower_input = torch.sum(tower_input, dim=0)
            tower_output = tower(tower_input)

            logit = linear(tower_output)
            output = out(logit)
            task_outs.append(output)

        # towers_input = [mmoe_out.t().unsqueeze(
        #     2).expand(-1, -1, self.expert_dnn_hidden_units[-1]) * expert_out for mmoe_out in mmoe_outs]
        # towers_input = [torch.sum(tower_input, dim=0)
        #                 for tower_input in towers_input]

        # # get the final output from the towers
        # towers_output = [tower(tower_input) for tower,
        #                  tower_input in zip(self.towers, towers_input)]

        # # get the output of the towers, and stack them
        # task_outs = torch.stack(towers_output, dim=1)

        return task_outs
