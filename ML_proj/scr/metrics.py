# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:04:06 2022

@author: waleed.al.haidri
"""

from keras import backend as K
import numpy as np
import tensorflow


def dice_soft_coef(y_true, y_pred, loss_type='jaccard', smooth=.1):
    y_pred = tensorflow.keras.backend.flatten(y_pred)
    y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

    y_true = tensorflow.keras.backend.flatten(y_true)
    y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)

    intersection = tensorflow.reduce_sum(y_true * y_pred)

    if loss_type == 'jaccard':
        union = tensorflow.reduce_sum(tensorflow.square(y_pred)) + tensorflow.reduce_sum(tensorflow.square(y_true))
    elif loss_type == 'sorensen':
        union = tensorflow.reduce_sum(y_pred) + tensorflow.reduce_sum(y_true)
    else:
        raise ValueError('Unknown `loss_type`: {}'.format(loss_type))

    return (2.0 * intersection + smooth) / (union + smooth)


def dice_coef(y_true, y_pred, smooth=0.01):
    # flatten
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
    y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 3)
    y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 3)
    # calculate intersection and union exluding background using y[:,1:]
    intersection = K.sum(y_true_f[:, 1:] * y_pred_f[:, 1:], axis=[-1])
    union = K.sum(y_true_f[:, 1:], axis=[-1]) + K.sum(y_pred_f[:, 1:], axis=[-1])
    # apply dice formula
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


