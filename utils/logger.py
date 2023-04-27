"""Logger."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import logging
import pprint


def get_logger(config, num_workers=1, rank=0):
    '''
    进行logger初始化
    :param config: 训练配置
    :param num_workers: 进程数
    :param rank: GPU序号
    :return: 配置完成的logger
    '''
    log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # noqa
    if num_workers > 1:
        log_file = '{}_{}_rank{}_{}.log'.format(
            config.utils.task_type,
            config.utils.task_name,
            rank,
            time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_file = '{}_{}_{}.log'.format(
            config.utils.task_type,
            config.utils.task_name,
            time.strftime('%Y-%m-%d-%H-%M'))
    log_path = os.path.join(config.utils.output_prefix, log_file)
    if log_path is not None:
        # mkdir
        if os.path.exists(log_path):
            os.remove(log_path)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # get logger
        logger = logging.getLogger()
        logger.handlers = []
        formatter = logging.Formatter(log_format)
        # file handler
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # stream handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # set level (info)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format=log_format)
    logger.info('\n############# Config #############\n{}'.format(
        pprint.pformat(config)))
    return logger
