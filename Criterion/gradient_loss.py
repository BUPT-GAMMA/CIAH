# -*- encoding: utf-8 -*-

#################################################################
# Implementation of Integrated Gradients function in Tensorflow #
# Naozumi Hiranuma (hiranumn@cs.washington.edu)                 #
#################################################################

import tensorflow as tf
import numpy as np
from typing import List

FLAGS = tf.flags.FLAGS

# 20210825 新加的
scale_for_grad = True


def gradient_guided_attn_module(grads_by_element_by_step: List[tf.Tensor], num_steps: int, attn: List[tf.Tensor]):
    """
    grads_by_element： list[ (E, F, D_f) ]
    num_steps：int
    attn： （E, C, F）
    """
    # integrated gradients
    stepsize = tf.convert_to_tensor(1. / num_steps)
    integrated_gradient_by_element = [grad * stepsize for grad in grads_by_element_by_step]
    integrated_gradient_by_element = tf.add_n(integrated_gradient_by_element)
    # element_wise_to_vector_wise
    integrated_gradient = tf.squeeze(tf.norm(integrated_gradient_by_element, axis=-1, ord="euclidean"), axis=0)  # (E, F')

    if scale_for_grad:
        with tf.variable_scope("scale", reuse=tf.AUTO_REUSE):
            scale_para = tf.get_variable("scale_para", shape=(), initializer=tf.ones_initializer(), )
        integrated_gradient = integrated_gradient * scale_para

    if FLAGS.gradguide.lower() == "l1":
        """ normalization by L1 (过于锐利，难以优化)"""
        normalized_integrated_gradient = tf.div_no_nan(integrated_gradient, tf.reduce_sum(integrated_gradient, axis=-1, keepdims=True))
    elif FLAGS.gradguide.lower() == 'stu':
        """ (logically doubt but useful) normalization similar as dual_cluster (larger means major here, while in dual cluster larger means minor ) """
        # integrated_gradient = 1. - tf.pow((1 / (1. + (integrated_gradient**2)) / FLAGS.stu_v), ((FLAGS.stu_v + 1.0) / 2.0))
        integrated_gradient = tf.pow(((1. + (integrated_gradient**2)) / FLAGS.stu_v), ((FLAGS.stu_v + 1.0) / 2.0))
        normalized_integrated_gradient = tf.div_no_nan(integrated_gradient, tf.reduce_sum(integrated_gradient, axis=1, keepdims=True))
    elif FLAGS.gradguide.lower() == 'minmax':
        """ normalization by min max scaled """
        min_value = tf.reduce_min(integrated_gradient, axis=-1, keepdims=True)
        max_value = tf.reduce_max(integrated_gradient, axis=-1, keepdims=True)
        normalized_integrated_gradient = tf.div_no_nan(integrated_gradient - min_value, max_value - min_value)
    elif FLAGS.gradguide.lower() == 'softmax':
        """ normalization by softmax """
        normalized_integrated_gradient = tf.nn.softmax(integrated_gradient, axis=-1)
    else:
        raise KeyError("Unknown gradguide: ", FLAGS.gradguide)

    # gradient guided loss
    normalized_integrated_gradient = tf.stop_gradient(normalized_integrated_gradient)
    ggl = tf.keras.losses.KLDivergence()(normalized_integrated_gradient, tf.reduce_sum(attn, axis=1))

    return ggl, normalized_integrated_gradient



def gradient_guided_attn_module_abandoned(grads_by_element: List[tf.Tensor], num_steps: int, attn: List[tf.Tensor]):
    """
    grads_by_element： list[ (E, F, D_f) ]
    num_steps：int
    attn： （E, C, F）
    """
    # prepare
    prepare_for_grad = lambda g: tf.squeeze(tf.norm(g, axis=-1, ord="euclidean"), axis=0)
    grads = [prepare_for_grad(grad) for grad in grads_by_element]  # list[(E, F')]
    attn = tf.reduce_sum(attn, axis=1)                             # (E, F')
    # integrated gradients
    integrated_gradient = []
    for grad in grads:
        stepsize = 1. / num_steps
        riemann = grad * stepsize
        integrated_gradient.append(riemann)
    integrated_gradient = tf.add_n(integrated_gradient)
    # gradient guided loss
    normalized_integrated_gradient = integrated_gradient / (tf.reduce_sum(integrated_gradient, axis=-1, keepdims=True) + 1e-9)
    normalized_integrated_gradient = tf.stop_gradient(normalized_integrated_gradient)
    ggl = tf.keras.losses.KLDivergence()(normalized_integrated_gradient, attn)

    return ggl, integrated_gradient


