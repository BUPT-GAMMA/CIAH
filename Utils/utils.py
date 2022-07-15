# -*- encoding: utf-8 -*-
import sys
import os
import tensorflow as tf

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if os.path.exists(filename):
            os.remove(filename)
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

    def change_file(self, filename="Default.log"):
        self.log.close()
        self.log = open(filename, "a")


def safe_lookup(paras, idx, name):
    emb = tf.nn.embedding_lookup(paras, idx, name=name)
    ret = tf.cond(tf.greater_equal(tf.cast(tf.reduce_max(idx), tf.int32), tf.shape(paras)[0]),
                  true_fn=lambda: tf.Print(tf.zeros_like(emb), [u"lookup out of range: " + name]),
                  false_fn=lambda: emb)
    return ret

def _sum_rows(x):
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])

def euclidean_dist(x, y, sqrt=True):
    ex = tf.expand_dims(x, axis=-2)
    ey = tf.expand_dims(y, axis=-3)
    distn = tf.norm(ex - ey + 1e-5, ord="euclidean", axis=-1)
    return distn


hyperpara_settings = {
        "acm": {
            "coef_dualcluster": 1., "coef_belong": 1.,  "coef_grad": 1.,    "coef_l2_emb": 0.1,
            "lr": 5e-3, "epoch": 1000,  "pretrain_epoch": 0, },
        "imd": {
            "coef_dualcluster": 1., "coef_belong": 1.,  "coef_grad": 50.,    "coef_l2_emb": 0.1,
            "lr": 5e-3, "epoch": 1000,  "pretrain_epoch": 500, },
        "mt4": {
            "coef_dualcluster": 0.1, "coef_belong": 1.,  "coef_grad": 10.,    "coef_l2_emb": 0.1,
            "lr": 2e-3, "epoch": 1000,  "pretrain_epoch": 200, },
        "mt9": {
            "coef_dualcluster": 0.1, "coef_belong": 1.,  "coef_grad": 20.,    "coef_l2_emb": 0.1,
            "lr": 2e-3, "epoch": 1000,  "pretrain_epoch": 200, },
    }



