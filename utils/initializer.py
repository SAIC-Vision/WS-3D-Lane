"""Initializer."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def initializer(init_type, x, init_params=None):
    '''
    pytorch模型初始化函数
    :param init_type: 初始化类型
    :param x: 被初始化的权重
    :param init_params: 初始化参数
    :return:
    '''
    if init_type.lower() == 'zero':
        out = torch.nn.init.zeros_(x)
    elif init_type.lower() == 'one':
        out = torch.nn.init.ones_(x)
    elif init_type.lower() == 'constant':
        out = torch.nn.init.constant_(x, val=init_params['value'])
    elif init_type.lower() == 'uniform':
        out = torch.nn.init.uniform_(x, a=init_params['scale'][0], b=init_params['scale'][1])
    elif init_type.lower() == 'normal':
        out = torch.nn.init.normal_(x, mean=init_params['mean'], std= init_params['std'])
    elif init_type.lower() == 'xavier_normal':
        out = torch.nn.init.xavier_normal_(x, gain=init_params['gain'])
    elif init_type.lower() == 'xavier_uniform':
        out = torch.nn.init.xavier_uniform_(x, gain=init_params['gain'])
    elif init_type.lower() == 'kaiming_normal':
        out = torch.nn.init.kaiming_normal_(x)
    elif init_type.lower() == 'kaiming_uniform':
        out = torch.nn.init.kaiming_uniform_(x)
    else:
        raise ValueError('Unknown initialization type {}.\n\
                          Please impletement it by yourself. ^_^'.format(init_type))  # noqa

    return out


