'''
评测函数
'''
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math

def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.

    preds : list of `NDArray`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
            labels = [labels]
            preds = [preds]

    return labels, preds


class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(``**config``)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):
        """Resets the local portion of the internal evaluation results
        to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            #print(self.num_inst)
            return (self.name, self.sum_metric / self.num_inst)

    def get_global(self):
        """Gets the current global evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self._has_global_stats:
            if self.global_num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.global_sum_metric / self.global_num_inst)
        else:
            return self.get()

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):
        """Returns zipped name and value pairs for global results.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()

class ClsAccuracy(EvalMetric):
    '''
    多分类准确度评价
    '''
    def __init__(self, config, eps=1e-5, name='Accuracy',
                 output_names=None, label_names=None, is_train=False):
        super(ClsAccuracy, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.config = config
        self.is_train = is_train

    def update(self, label, pred):

        label = label.squeeze().cpu().numpy()
        pred = pred.squeeze()
        pred = F.softmax(pred, dim=-1)
        pred = pred.detach().cpu().numpy()
        label = np.int64(label.ravel())
        pred_cls = np.argmax(pred, axis=1)

        self.sum_metric += np.sum((label == pred_cls))
        self.num_inst += np.sum(label.shape[0])


class AngleMAE(EvalMetric):
    '''
    多分类准确度评价
    '''
    def __init__(self, config, eps=1e-5, name='AngleMAE',
                 output_names=None, label_names=None, is_train=False):
        super(AngleMAE, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.config = config
        self.is_train = is_train

    def update(self, label, pred, batch_center_line_lst):

        label = label.squeeze().cpu().numpy()
        pred = pred.squeeze()
        pred = pred.detach().cpu().numpy()

        no_use_idx = []
        for idx in range(len(batch_center_line_lst)):
            if batch_center_line_lst[idx][0][0][0] == batch_center_line_lst[idx][1][0][0]:
                pred[idx] = 0
                label[idx] = 0
                no_use_idx.append(idx)
        #print(no_use_idx)

        self.sum_metric += np.sum((abs(label - pred) * 180 / math.pi))
        self.num_inst += (np.sum(label.shape[0]) - len(no_use_idx))





