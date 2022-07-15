# -*- encoding: utf-8 -*-
import tensorflow as tf
from Utils import euclidean_dist

FLAGS = tf.flags.FLAGS
# cross = FLAGS.cross

target_update_step_num = 1

def target_distribution(q):
    weight = q**2 / tf.reduce_sum(q, axis=0)
    return tf.stop_gradient(weight / tf.reduce_sum(weight, axis=1, keepdims=True), name="p")

def indexing_second(fea, idx):
    rc_index = tf.stack([tf.range(0, tf.shape(fea)[0]), idx], axis=1)
    ans = tf.gather_nd(fea, rc_index)
    return ans

def cluster_p_and_q(embeddings, cluster_centers, v, return_inertia):
    """
    embeddings:         (E, D)
    cluster_centers:    (C, D)
    v: int
    """
    if FLAGS.dist == "euclidean":
        # Distance_{ij}^2 = ||X_i||^2 + ||Y_j||^2 - 2 \cdot X_i^T \cdot Y_j
        distance = euclidean_dist(embeddings, cluster_centers) ** 2
        pred = tf.pow((1. / (1. + distance / v)), ((v + 1.0) / 2.0))  # X^(-(1+v)/2)
        pred = pred / tf.reduce_sum(pred, axis=1, keepdims=True)  # 这一句：  计算结果对pred的导数，为0
    elif FLAGS.dist == "innerproduct":
        distance = tf.matmul(embeddings, cluster_centers, transpose_b=True)         # (E, C)
        pred = tf.nn.softmax(distance, axis=1)
    else:
        return KeyError("Unrecognized FLAGS.dist in emb_weight_dual_cluster_module, ", FLAGS.dist)

    target = target_distribution(pred)

    if return_inertia:
        near_distance = indexing_second(distance, tf.argmax(pred, axis=1, output_type=tf.int32))
        far_distance = indexing_second(distance, tf.argmin(pred, axis=1, output_type=tf.int32))
        inertia_rate = tf.reduce_mean(tf.div_no_nan(near_distance, far_distance))
        return target, pred, inertia_rate
    else:
        return target, pred, None




# Dual Self-supervised Module
def emb_weight_dual_cluster_module(batch_size, num_classes, emb, emb_cluster_center,
                                   weight, weight_cluster_center, global_step,
                                   dropout, training, v=1, return_inertia=False):
    """
    emb:                    (E, D)      or (E, C, D)
    emb_cluster_center:     (C, D)      or (C, D * C)
    weight:                 (E, 1, D)   or (E, C, D)
    weight_cluster_center:  (C, D)      or (C, D * C)
    """
    with tf.variable_scope("cluster_loss"):
        # batch_size = tf.shape(emb)[0]    # E
        target_wgt = tf.get_variable("target_wgt", shape=[batch_size, num_classes],
                                     initializer=tf.constant_initializer(value=1/num_classes), trainable=False)
        target_emb = tf.get_variable("target_emb", shape=[batch_size, num_classes],
                                     initializer=tf.constant_initializer(value=1/num_classes), trainable=False)

        # dropoutlayer = tf.layers.Dropout(rate=dropout)
        if FLAGS.attn == 'channel' and FLAGS.embedding_model != "SceneMiningMultiHotModel":
            # C > 1
            raise NotImplementedError
            # classifier = tf.layers.Dense(units=1, name=)
            # predict = classifier(dropoutlayer(emb, training=training))
            # predict = tf.nn.softmax(tf.squeeze(predict, axis=-1))
            # emb = tf.reshape(emb, [batch_size, -1])
        else:
            # C = 1
            weight = tf.squeeze(weight, axis=1)   # (E, D)
            new_target_wgt, pred_wgt, _ = cluster_p_and_q( weight, weight_cluster_center, v=v, return_inertia=False)

        new_target_emb, pred_emb, inertia = cluster_p_and_q( emb, emb_cluster_center, v=v, return_inertia=return_inertia )

        target_wgt = tf.cond(tf.equal(global_step % target_update_step_num, 0), lambda: target_wgt.assign(new_target_wgt), lambda: target_wgt)
        target_emb = tf.cond(tf.equal(global_step % target_update_step_num, 0), lambda: target_emb.assign(new_target_emb), lambda: target_emb)

        if FLAGS.cluster_type == 'cross':
            # 交换了目标分布
            cluster_emb_loss = tf.keras.losses.KLDivergence(name="cluster_emb_loss")(target_wgt, pred_emb)
            cluster_wgt_loss = tf.keras.losses.KLDivergence(name="cluster_wgt_loss")(target_emb, pred_wgt)
            cluster_loss = 1 * cluster_wgt_loss + 1 * cluster_emb_loss

        elif FLAGS.cluster_type == 'dual':
            cluster_emb_loss = tf.keras.losses.KLDivergence(name="cluster_emb_loss")(target_emb, pred_emb)
            cluster_wgt_loss = tf.keras.losses.KLDivergence(name="cluster_wgt_loss")(target_wgt, pred_wgt)
            cluster_crs_loss = tf.keras.losses.KLDivergence(name="cluster_crs_loss1")(pred_wgt, pred_emb) + \
                               tf.keras.losses.KLDivergence(name="cluster_crs_loss2")(pred_emb, pred_wgt)
            cluster_loss = 1 * cluster_wgt_loss + 1 * cluster_emb_loss + 1 * cluster_crs_loss

        elif FLAGS.cluster_type == 'single':
            cluster_emb_loss = tf.keras.losses.KLDivergence(name="cluster_emb_loss")(target_emb, pred_emb)
            cluster_loss = 1 * cluster_emb_loss


        # cla_loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # labels = tf.cast(pred_emb == tf.reduce_max(pred_emb, axis=1, keepdims=True), tf.float32)
        # loss_cla = cla_loss_func(y_true=labels, y_pred=pred_emb)
        # cluster_loss += 0.01 * loss_cla

        return cluster_loss, target_emb, pred_emb, inertia

