# -*- encoding: utf-8 -*-
import tensorflow as tf

from .simple_attention import simple_attention
from .utils import get_shape

flags = tf.flags
FLAGS = flags.FLAGS
if FLAGS.dataset.lower() == 'acm':
    num_class = 3
elif FLAGS.dataset.lower() == 'imdb':
    num_class = 3
elif FLAGS.dataset.lower() == 'mt4':
    num_class = 4
elif FLAGS.dataset.lower() == 'mt9':
    num_class = 9

def attn(Q, K, Q_idx_in_K, K_mask, dropout, training, scope=None):
    """
    :param Q:          (B, TQ, D)   query: 自身节点/超边表示, or behavior表示
    :param K:          (B, TK, D)   key: 自身节点/超边表示拼接邻居(超边/节点)
    :param Q_idx_in_K：(B, T , D)   自身表示在 K 中所占位置（这里不考虑Q中的padding）
    :param K_mask：    (B, TK, D)   K 中含有padding，padding不应被融合
    """
    if FLAGS.attn == 'simple' or FLAGS.attn == "simiter":
        with tf.variable_scope(scope or "simpleattn"):
            [outputs, inputs_mat], alpha_weight = simple_attention(K, K_mask, dropout, training, att_dim=8, scope=scope) # (B, D)

            outputs, alpha_weight = tf.expand_dims(outputs, axis=1), tf.expand_dims(alpha_weight, axis=1)

            if 'layer1' in tf.get_variable_scope().name:
                if FLAGS.norm.lower() == 'ln':
                    outputs = tf.contrib.layers.layer_norm(inputs=outputs, begin_norm_axis=-1, begin_params_axis=-1,
                                                           scope="layer_norm", reuse=tf.AUTO_REUSE)
                elif FLAGS.norm.lower() == 'bn':
                    outputs = tf.contrib.layers.batch_norm(inputs=outputs, center=False, scale=True,
                                                           is_training=training, scope="layer_norm", reuse=tf.AUTO_REUSE)
                elif FLAGS.norm.lower() == 'none':
                    outputs = outputs
                else:
                    raise KeyError("Unknown parameter of norm: ", FLAGS.norm)

            if FLAGS.simple_keepdim:
                return [outputs, inputs_mat], alpha_weight
            else:
                return [outputs, outputs], alpha_weight   # (B, 1, D), (B, 1, TK)

    else:
        raise KeyError("Unknown attention module: {}. ".format(FLAGS.attn))
