# -*- encoding: utf-8 -*-
from typing import List
from FeatureSelection import attn
import tensorflow as tf
from Utils import safe_lookup

FLAGS = tf.flags.FLAGS

def SubLayer_N2E(E4N_list: List, feat_list: List, mask_feat_list: List,
                 grad_alpha: tf.placeholder,
                 dropout=0.5, training=True, scope=None, grad=False):
    """
    self || neighborNodes = e || n^e_1 || n^e_2 || ... || n^e_{N_t}   =>  e

    E4N_list: a list of tensors of shape (E, N_1), ..., (E, N_t), ...
    feat_list: a list of tensors of shape (E, F_E, d), ..., (N_t, F_Nt, d), ...
    mask_feat_list: a list of tensors of shape (E, F_E), ..., (N_t, F_Nt), ...
    """
    with tf.variable_scope(scope or "sublayer_n2e", reuse=tf.compat.v1.AUTO_REUSE):
        # concat neighbors
        E_repr, Ns_repr = feat_list[0], feat_list[1: ]                             # (E, F_E, d)
        E_mask, Ns_mask = mask_feat_list[0], mask_feat_list[1: ]
        E_mask_with_pad = tf.cast(tf.range(tf.shape(E_repr)[1]), dtype=tf.int64)   # 0~F_E
        neighbor_concat, mask_concat = [E_repr], [E_mask]
        for t in range(len(E4N_list)):
            nghb_t = safe_lookup(Ns_repr[t], E4N_list[t], name="nghb_{}".format(t))      # (E, N_t, F_Nt, d)
            mask_t = safe_lookup(Ns_mask[t], E4N_list[t], name="mask_{}".format(t))      # (E, N_t, F_Nt)
            dims = tf.shape(nghb_t)                                                                 # (E, N_t, F_Nt, d)
            neighbor_concat.append(tf.reshape(nghb_t, shape=(dims[0], dims[1]*dims[2], dims[3])))   # (E, N_t * F_Nt, d)
            mask_concat.append(tf.reshape(mask_t, shape=(dims[0], dims[1]*dims[2])))                # (E, N_t * F_Nt)

        neighbor_concat = tf.concat(neighbor_concat, axis=-2)                         # (E, F_E + \sum_t{N_t * F_Nt}, d)
        mask_concat = tf.concat(mask_concat, axis=-1)                                 # (E, F_E + \sum_t{N_t * F_Nt})

        # attention for aggregate features of all neighbors
        # (E, 1, d), (E, F_E, d), (E, F_E + \sum_t{N_t * F_Nt})
        # Q, K, Q_idx_in_K, K_mask, dropout, training, scope=None
        (vec_output, mat_output), weights = attn(E_repr * grad_alpha, neighbor_concat * grad_alpha,
                                                 E_mask_with_pad, mask_concat,
                                                 dropout=dropout, training=training)

        return (vec_output, mat_output), (neighbor_concat, weights)   # (representation of E by vec, mat), (input of attn, weight of attn)



def SubLayer_E2N(N4E_list: List, feat_list: List, mask_feat_list: List,
                 dropout=0.5, training=True, scope=None):
    """
    self || neighborEdges = n_t || e_1 || e_2 || ... || e_{N_t}   =>  n_t  t=1,2,...

    E4N_list: a list of tensors of shape (N_1, E_1), ..., (N_t, E_t), ...
    feat_list: a list of tensors of shape (E, F_E, d), ..., (N_t, F_Nt, d), ...
    mask_feat_list: a list of tensors of shape (E, F_E), ..., (N_t, F_Nt), ...
    """
    vec_outputs, mat_outputs, weights = [], [], []
    with tf.variable_scope(scope or "sublayer_e2n"):
        E_repr, E_mask = feat_list[0], mask_feat_list[0]                                 # (E, F_E, d)
        for t in range(len(N4E_list)):
            Nt_repr, Nt_mask = feat_list[1 + t], mask_feat_list[1 + t]                   # (N_t, F_Nt, d), (N_t, F_Nt)
            Nt_mask_with_pad = tf.cast(tf.range(tf.shape(Nt_repr)[1]), dtype=tf.int64)   # 0~F_Nt
            e_nghbs_t = safe_lookup(E_repr, N4E_list[t], name="e_nghbs_{}".format(t))    # (N_t, E_t, F_E, d)
            mask_t = safe_lookup(E_mask, N4E_list[t], name="mask_{}".format(t))          # (N_t, E_t, F_E)
            dims = tf.shape(e_nghbs_t)                                                      # (N_t, E_t, F_E, d)
            e_nghbs_t = tf.reshape(e_nghbs_t, shape=(dims[0], dims[1]*dims[2], dims[3]))    # (N_t, E_t * F_E, d)
            mask_t = tf.reshape(mask_t, shape=(dims[0], dims[1]*dims[2]))

            neighbor_concat = tf.concat([Nt_repr, e_nghbs_t], axis=-2)              # (N_t, F_Nt + E_t * F_E, d)
            mask_concat = tf.concat([Nt_mask, mask_t], axis=-1)                     # (N_t, F_Nt + E_t * F_E)

            # attention for aggregate features of all neighboring edges (and self)
            # (N_t, 1, d), (N_t, F_Nt, d), (N_t, F_Nt + E_t * F_E)
            (vec_output_t, mat_output_t), weight_t = attn(
                                                Nt_repr, neighbor_concat,
                                                Nt_mask_with_pad, mask_concat,
                                                dropout=dropout, training=training, scope="Nt_attn_{}".format(t))

            vec_outputs.append(vec_output_t)
            mat_outputs.append(mat_output_t)
            weights.append(weight_t)

        return (vec_outputs, mat_outputs), weights

