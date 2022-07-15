# -*- encoding: utf-8 -*-
import tensorflow as tf
from Utils import _sum_rows, euclidean_dist
FLAGS = tf.flags.FLAGS

def inner_product_decoder(input_emb1, input_emb2, activation, dropout, training):
    dropoutlayer = tf.layers.Dropout(rate=dropout)
    input1 = dropoutlayer(input_emb1, training=training)                                # (E, D)
    input2 = dropoutlayer(input_emb2, training=training)                                # (N, D)
    inner_product = tf.matmul(input1, input2, transpose_b=True)                         # (E, N)
    outputs = activation(inner_product)
    return outputs

def lil_sparse_belonging_list_to_dense(belonging_matrix, row_num, col_num):
    """
    belonging_matrix: (E, n)    n < N
    row_num: E
    col_num: N
    """
    col_idx = belonging_matrix          # (E, n)
    row_idx = tf.reshape(tf.cast(tf.range(row_num), tf.int64), [-1, 1])                         # (E, 1)
    row_idx = tf.tile(row_idx, multiples=[1, tf.shape(belonging_matrix)[-1]])                   # (E, n)
    indices = tf.reshape(tf.stack([row_idx, col_idx], axis=-1), [-1, 2])                        # (E*n, 2)
    values =  tf.reshape(tf.ones_like(belonging_matrix, dtype=tf.float32), [-1])                # (E*n, )

    sparse = tf.SparseTensor(indices=indices, values=values, dense_shape=[row_num, col_num])
    dense = tf.sparse.to_dense(sparse, validate_indices=False)
    return dense

def node_belonging_to_hyperedge_module(behavior_embedding, entity_embeddings, belonging_list, node_num_list,
                                       dropout, training):
    if tf.rank(behavior_embedding) == 3:    # 若有channel，则拼接为一排
        D = tf.shape(behavior_embedding)[-1]
        behavior_embedding = tf.reshape(behavior_embedding, [-1, D])
        entity_embeddings = tf.reshape(entity_embeddings, [-1, D])

    losses_for_each_type = []
    for n_type in range(len(entity_embeddings)):
        if 'mt' in FLAGS.dataset and n_type == 2:
            continue
        belonging_preds = inner_product_decoder(
            behavior_embedding, entity_embeddings[n_type],
            activation=lambda X: X, dropout=dropout, training=training)
        belonging_truel = lil_sparse_belonging_list_to_dense(
            belonging_list[n_type],
            row_num=node_num_list[0], col_num=node_num_list[1+n_type])
        mse_loss = tf.keras.losses.MeanSquaredError()
        losses_for_each_type.append(mse_loss(y_true=belonging_truel, y_pred=belonging_preds))

    return tf.add_n(losses_for_each_type)


def nce_inner_product(behavior_embedding, entity_embedding, belonging_labels, candidate_num, positive_num, negative_sampled_num, name):
    negative_sampled_num = min(negative_sampled_num, candidate_num)
    negative_sampler = tf.random.uniform_candidate_sampler(
        true_classes=belonging_labels,
        num_true=positive_num,
        num_sampled=negative_sampled_num,
        unique=True,  # True 会溢出
        range_max=candidate_num,
        name=name
    )
    bias_for_logits = tf.get_variable(
        "nce_bias_" + str(name),
        shape=[candidate_num, ],
        # trainable=False,
        initializer=tf.zeros_initializer()
    )
    nce_loss_ip = tf.nn.nce_loss(
        weights=entity_embedding,
        biases=bias_for_logits,
        labels=tf.cast(belonging_labels, tf.int64),
        inputs=behavior_embedding,
        num_sampled=negative_sampled_num,
        num_classes=candidate_num,
        num_true=positive_num,
        sampled_values=negative_sampler,
        remove_accidental_hits=True,
        name="nce_{}".format(name),
    )
    return nce_loss_ip, None


def nce_euclidean_distance(behavior_embedding, entity_embedding, belonging_labels, candidate_num, positive_num, negative_sampled_num, name):
    negative_sampled_num = min(negative_sampled_num, candidate_num)
    negative_sampler = tf.random.uniform_candidate_sampler(
        true_classes=belonging_labels,
        num_true=positive_num,
        num_sampled=negative_sampled_num,
        unique=True,  # True 会溢出
        range_max=candidate_num
    )
    """
    Modified based on tf codes in nce_loss. 
    """
    with tf.name_scope("calculates_logits"):
        labels_flat = tf.reshape(tf.cast(belonging_labels, tf.int64), [-1])
        sampled, true_expected_count, sampled_expected_count = (tf.stop_gradient(s) for s in negative_sampler)
        sampled = tf.cast(sampled, tf.int64)
        all_ids = tf.concat([labels_flat, sampled], 0)
        # weights shape is [num_classes, dim], true_w shape is [batch_size * num_true, dim]
        all_w = tf.nn.embedding_lookup(entity_embedding, all_ids)
        true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
        sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
        # inputs shape is [batch_size, dim], true_w shape is [batch_size * num_true, dim], row_wise_dots is [batch_size, num_true, dim]
        dim = tf.shape(true_w)[1:2]
        new_true_w_shape = tf.concat([[-1, positive_num], dim], 0)   # [batch_size, num_true, dim]
        # We want the row-wise dot plus biases which yields a [batch_size, num_true] tensor of true_logits.
        sampled_logits = euclidean_dist(behavior_embedding, sampled_w)                                              # euclidean distance
        true_logits = euclidean_dist(tf.expand_dims(behavior_embedding, 1), tf.reshape(true_w, new_true_w_shape))   # euclidean distance
        true_logits = tf.reshape(true_logits, [-1, positive_num], name="true_logits")

    """ remove accidental hits: """
    with tf.name_scope("remove_accidental_hits"):
        acc_indices, acc_ids, acc_weights = tf.nn.compute_accidental_hits(belonging_labels, sampled, num_true=positive_num)
        # This is how SparseToDense expects the indices.
        acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
        acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
        sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1, "sparse_indices")
        # Create sampled_logits_shape = [batch_size, num_sampled]
        sampled_logits_shape = tf.concat([tf.shape(belonging_labels)[:1], tf.expand_dims(negative_sampled_num, 0)], 0)
        if sampled_logits.dtype != acc_weights.dtype:
            acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
        # sampled_logits += tf.sparse_to_dense(
        sampled_logits -= tf.sparse_to_dense(
            sparse_indices,
            sampled_logits_shape,
            acc_weights,
            default_value=0.0,
            validate_indices=False)

    with tf.name_scope("loss"):
        margin = 1
        contrastive_pos = 1 / 2 * tf.reduce_mean((tf.pow(true_logits, 2)))
        contrastive_neg = 1 / 2 * tf.reduce_sum(tf.pow(tf.nn.relu(margin - sampled_logits), 2))
        contrastive_neg = tf.div_no_nan(
            contrastive_neg,
            tf.cast(tf.shape(belonging_labels)[0] * negative_sampled_num - tf.shape(acc_weights)[0], tf.float32)
        )
        with tf.control_dependencies([
            tf.cond(tf.greater(contrastive_pos + contrastive_neg, 100), lambda: tf.print(contrastive_pos, contrastive_neg), lambda: tf.print(end=""))
        ]):
            contrastive_loss = contrastive_pos + contrastive_neg
        # bpr_p = tf.reduce_mean(tf.exp( - 1 / 2 * true_logits))
        # bpr_n = tf.reduce_sum(tf.exp( - 1 / 2 * sampled_logits))
        # bpr_n = tf.div_no_nan(bpr_n, tf.cast(tf.shape(belonging_labels)[0] * negative_sampled_num - tf.shape(acc_weights)[0], tf.float32))
        # bpr_loss = - tf.log(tf.sigmoid(bpr_p - bpr_n))

        loss = contrastive_loss
    return loss, (contrastive_pos, contrastive_neg)


def node_belonging_to_hyperedge_NCEloss_module(behavior_embedding, entity_embeddings, belonging_list, node_num_list,
                                       dropout, training):
    negative_sampled_num = 64
    with tf.variable_scope("NCE_Loss"):
        if tf.rank(behavior_embedding) == 3:    # 若有channel，则拼接为一排
            behavior_embedding = tf.reshape(behavior_embedding, [tf.shape(behavior_embedding)[0], -1])
            entity_embeddings = tf.reshape(entity_embeddings, [tf.shape(entity_embeddings)[0], -1])

        losses_for_each_type = []
        pos_loss_for_each_type, neg_loss_for_each_type = [tf.zeros(())], [tf.zeros(())]
        for n_type in range(len(entity_embeddings)):
            if 'mt' in FLAGS.dataset and n_type == 2:
            # if 'mt' in FLAGS.dataset and n_type >= 1:
                continue
            if FLAGS.dist == 'innerproduct':
                nce_loss, _ = nce_inner_product(
                    behavior_embedding=behavior_embedding,
                    entity_embedding=entity_embeddings[n_type],
                    belonging_labels=belonging_list[n_type],
                    candidate_num=node_num_list[1 + n_type],
                    positive_num=belonging_list[n_type].get_shape()[1].value,
                    negative_sampled_num=negative_sampled_num,
                    name=str(n_type)
                )
            elif FLAGS.dist == 'euclidean':
                nce_loss, (pos_loss, neg_loss) = nce_euclidean_distance(
                    behavior_embedding=behavior_embedding,
                    entity_embedding=entity_embeddings[n_type],
                    belonging_labels=belonging_list[n_type],
                    candidate_num=node_num_list[1 + n_type],
                    positive_num=belonging_list[n_type].get_shape()[1].value,
                    negative_sampled_num=negative_sampled_num,
                    name=str(n_type)
                )
                pos_loss_for_each_type.append(tf.reduce_mean(pos_loss))
                neg_loss_for_each_type.append(tf.reduce_mean(neg_loss))
            else:
                raise KeyError("Unrecognized FLAGS.dist in node_belonging_to_hyperedge_NCEloss_module, ", FLAGS.dist)

            losses_for_each_type.append(tf.reduce_mean(nce_loss))

        return tf.add_n(losses_for_each_type), (tf.add_n(pos_loss_for_each_type), tf.add_n(neg_loss_for_each_type))
