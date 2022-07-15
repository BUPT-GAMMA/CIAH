import tensorflow as tf
from .baseModel import BaseModel
from GNN import SubLayer_E2N, SubLayer_N2E
from Utils import safe_lookup

FLAGS = tf.flags.FLAGS
shape_changed_attns = ["simple", 'channel', "simiter"]
linear_for_weight = True
linear_for_weight_share = True
block = False

class SceneMiningBaselineModel(BaseModel):
    def __init__(self,
                 input_data,
                 embedding_size,
                 grad_alpha = 1.,
                 silent_print=False
                 ):
        super(SceneMiningBaselineModel, self).__init__(input_data, embedding_size, silent_print)

        self.layer_num = FLAGS.layer
        self.init_hypergraph_neural_network(grad_alpha)


    def init_hypergraph_neural_network(self, grad_alpha):    # grad_alpha: scalar
        grad = (FLAGS.coef_grad > 0)
        self.first_attn_inputs = None

        with tf.variable_scope('hgnn'):
            self.hyperedge_vec_list, self.hyperedge_mat_list = [], []
            self.node_vec_list, self.node_mat_list = [], []
            self.sub1_attn_weight_list, self.sub2_attn_weight_list = [], []

            # middle layers
            features = [self.hyperedge_init_embedding] + self.nodes_init_embedding
            mask_feats = self.mask_feats
            for layer in range(1, self.layer_num ):
                # TODO: 要想通用化attn值和索引的映射，这里 hyperedge_mat 在 simple 的输出，应加一行0，这样可以保持第一行是padding的设定
                sub1_output = SubLayer_N2E(self.E4N, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                (hyperedge_vec, hyperedge_mat), (attn_inputs, attn_weights) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs
                grad = False
                # edge feature 更新了
                features = [hyperedge_mat] + features[1: ]
                new_edge_mask_feats = tf.cast(tf.ones(tf.shape(hyperedge_mat)[:-1]), tf.bool) \
                                        if (FLAGS.attn in ["simple", "simiter"] and not FLAGS.simple_keepdim) \
                                        else mask_feats[0]
                mask_feats = [new_edge_mask_feats] + mask_feats[1:]
                self.hyperedge_vec_list.append(hyperedge_vec)
                self.hyperedge_mat_list.append(hyperedge_mat)
                self.sub1_attn_weight_list.append(attn_weights)
                sub2_output = SubLayer_E2N(self.N4E, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub2_E2N".format(layer))
                (node_vecs, node_mats), attn_weights = sub2_output
                # node feature 更新了
                features = features[:1] + node_mats
                new_node_mask_feats = [
                    (tf.cast(tf.ones(tf.shape(node_mat)[:-1]), tf.bool)
                        if (FLAGS.attn in ["simple", "simiter"] and not FLAGS.simple_keepdim) else mask_feats[i+1])
                    for i, node_mat in enumerate(node_mats)
                ]
                mask_feats = mask_feats[:1] + new_node_mask_feats
                self.node_vec_list.append(node_vecs)
                self.node_mat_list.append(node_mats)
                self.sub2_attn_weight_list.append(attn_weights)


            # last layer
            layer = self.layer_num
            if FLAGS.coef_reconst > 0:
                E4N_list = [tf.concat([self.E4N[t], self.neg_E4N_list[t]], axis=0) for t in range(len(self.E4N))]
                self.neg_E_feature = tf.tile(features[0], [FLAGS.negnum, 1, 1])  # 顺序不变
                features = [tf.concat([features[0], self.neg_E_feature], axis=0)] + features[1: ]
                neg_E_mask_feats = tf.tile(mask_feats[0], [FLAGS.negnum, 1])  # 顺序不变
                cache_mask_feats = [tf.concat([mask_feats[0], neg_E_mask_feats], axis=0)] + mask_feats[1: ]
                sub1_output = SubLayer_N2E(E4N_list, features, cache_mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                grad = False
                (hyperedge_vec_pos_and_neg, hyperedge_mat_pos_and_neg), (attn_inputs_pos_and_neg, attn_weights_pos_and_neg) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs_pos_and_neg

                self.reconstruct_embd = tf.squeeze(hyperedge_vec_pos_and_neg, axis=1)
                self.reconstruct_label = tf.concat([tf.ones(self.pos_num), tf.zeros(self.pos_num * FLAGS.negnum)], axis=0)

                hyperedge_vec = hyperedge_vec_pos_and_neg[: self.pos_num]
                hyperedge_mat = hyperedge_mat_pos_and_neg[: self.pos_num]
                attn_weights = attn_weights_pos_and_neg[: self.pos_num]

            else:
                sub1_output = SubLayer_N2E(self.E4N, features, mask_feats,
                                           dropout=self.dropout, training=self.training,
                                           scope="layer{}_sub1_N2E".format(layer),
                                           grad_alpha=grad_alpha if grad else 1.)
                grad = False
                (hyperedge_vec, hyperedge_mat), (attn_inputs, attn_weights) = sub1_output
                if self.first_attn_inputs is None:
                    self.first_attn_inputs = attn_inputs

            # edge feature 更新了
            features = [hyperedge_mat] + features[1: ]
            new_edge_mask_feats = tf.cast(tf.ones(tf.shape(hyperedge_mat)[:-1]), tf.bool) \
                if (FLAGS.attn in shape_changed_attns) \
                else mask_feats[0]
            mask_feats = [new_edge_mask_feats] + mask_feats[1:]
            self.hyperedge_vec_list.append(hyperedge_vec)
            self.hyperedge_mat_list.append(hyperedge_mat)
            self.sub1_attn_weight_list.append(attn_weights)
            sub2_output = SubLayer_E2N(self.N4E, features, mask_feats,
                                       dropout=self.dropout, training=self.training,
                                       scope="layer{}_sub2_E2N".format(layer))
            (node_vecs, node_mats), attn_weights = sub2_output
            # node feature 更新了
            features = features[:1] + node_mats
            new_node_mask_feats = [
                (tf.cast(tf.ones(tf.shape(node_mat)[:-1]), tf.bool)
                 if (FLAGS.attn in shape_changed_attns) else mask_feats[i + 1])
                for i, node_mat in enumerate(node_mats)
            ]
            mask_feats = mask_feats[:1] + new_node_mask_feats
            self.node_vec_list.append(node_vecs)
            self.node_mat_list.append(node_mats)
            self.sub2_attn_weight_list.append(attn_weights)

            if FLAGS.layer_aggr == 'concat':
                self.behavior_emb_concat = tf.concat(self.hyperedge_vec_list, axis=-1, name="behavior_emb")  # (E, 1 or C, D)
                self.entity_emb_concats = [tf.concat(te, axis=-1, name="entity_{}_emb".format(t))
                                           for t, te in enumerate(zip(*self.node_vec_list))]  # (N_t, 1 or C, D)

            else:
                self.behavior_emb_concat = sum(self.hyperedge_vec_list)  # (E, 1 or C, D)
                self.entity_emb_concats = [sum(te) for t, te in enumerate(zip(*self.node_vec_list))]  # (N_t, 1 or C, D)

            if FLAGS.attn == 'channel':
                pass
            else:
                self.behavior_emb_concat = tf.squeeze(self.behavior_emb_concat, axis=1)  # (E, D)
                self.entity_emb_concats = [tf.squeeze(te, axis=1) for te in self.entity_emb_concats]  # (N_t, D)

    def logits(self, entity=False):
        if entity:
            return self.behavior_emb_concat, self.entity_emb_concats
        else:
            return self.behavior_emb_concat

    def reconstruct_emb_and_label(self):
        return self.reconstruct_embd, self.reconstruct_label

    def attn_weights(self):
        with tf.variable_scope("get_attn_weights"):
            if FLAGS.attn in ['simple', 'channel', 'simiter']:
                C = 1 if FLAGS.attn.startswith('sim') else self.num_classes
                sub1_weights, sub2_weights = [], []
                self.target_layer = 0
                if self.target_layer != 0 and FLAGS.attn.startswith('sim'):
                    raise NotImplementedError
                # sub layer 1: E4N (N2E)
                weights = self.inverse_attn_or_gradient(self.sub1_attn_weight_list[self.target_layer])  # 先只计算第一层
                self.ori_attn = weights

                if block:
                    start = 1
                    end = 1904 if FLAGS.dataset == 'acm' else 999999
                    weights = weights[:, :, start: end]
                    weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)

                # ↑ 第一列是特征padding，为保持embedding的纯粹，这里不移除
                if linear_for_weight:
                    # linear:
                    name_post = "" if linear_for_weight_share else "_sub1"
                    self.linear_layer_for_weight = tf.layers.Dense(self.embedding_size, activation=tf.nn.tanh,
                                                                   name="linear_for_weight"+name_post, _reuse=tf.AUTO_REUSE)
                    weights = self.linear_layer_for_weight(weights)
                sub1_weights.append(weights)

                # sub layer 2: N4E (E2N)
                attn = self.sub2_attn_weight_list[self.target_layer]   # 先只计算第一层
                weights = []
                for t in range(self.num_node_types):
                    N_t_num = self.num_nodes[t+1]
                    col_idx_t_n = tf.tile(tf.reshape(self.features[t+1], [N_t_num, 1, -1]), multiples=[1, C, 1]) # (N_t, C, F_Nt)
                    if FLAGS.attn.startswith('sim'):
                        if self.target_layer == 0:
                            col_idx_t_e = safe_lookup(
                                tf.zeros(self.hyperedge_mat_list[self.target_layer].shape[:-1], dtype=tf.int64),
                                self.N4E[t],
                                "sub2_col_idx_{}".format(t))   # (N_t, E_t, 1)
                        else:
                            raise NotImplementedError
                    else:
                        col_idx_t_e = safe_lookup(self.features[0], self.N4E[t], "sub2_col_idx_{}".format(t))   # (N_t, E_t, F_E)
                    col_idx_t_e = tf.tile(tf.reshape(col_idx_t_e, [N_t_num, 1, -1]), multiples=[1, C, 1])   # (N_t, C, E_t*F_E)

                    col_idx = tf.concat([col_idx_t_n, col_idx_t_e], axis=-1)  # (N_t, C, F_Nt+E_t*F_E)
                    row_idx = tf.reshape(tf.cast(tf.range(N_t_num), tf.int64), [-1, 1, 1])  # (N_t, 1, 1)
                    row_idx = tf.tile(row_idx, multiples=[1, C, tf.shape(col_idx)[-1]])  # (N_t, C, F_Nt+E_t*F_E)
                    chn_idx = tf.reshape(tf.cast(tf.range(C), tf.int64), [1, -1, 1])  # (1, C, 1)
                    chn_idx = tf.tile(chn_idx, multiples=[N_t_num, 1, tf.shape(col_idx)[-1]])  # (N_t, C, F_Nt+E_t*F_E)
                    indices = tf.reshape(tf.stack([row_idx, chn_idx, col_idx], axis=-1), [-1, 3])
                    values = tf.reshape(attn[t], [-1])  # (N_t * C * (F_Nt+E_t*F_E))

                    weights_t = tf.SparseTensor(indices=indices, values=values, dense_shape=[N_t_num, C, self.feature_vocab_size])
                    weights_t = tf.sparse.to_dense(weights_t, validate_indices=False)

                    if block:
                        start = 1
                        end = 1904 if FLAGS.dataset == 'acm' else 999999
                        weights_t = weights_t[:, :, start: end]
                        weights_t = weights_t / tf.reduce_sum(weights_t, axis=-1, keepdims=True)

                    if linear_for_weight:
                        # linear:
                        name_post = "" if linear_for_weight_share else ("_sub2_"+str(t))
                        linear_layer_t = tf.layers.Dense(self.embedding_size, activation=tf.nn.tanh,
                                                         name="linear_for_weight"+name_post, _reuse=tf.AUTO_REUSE)
                        weights_t = linear_layer_t(weights_t)

                    weights.append(weights_t)

                sub2_weights.append(weights)

                return sub1_weights, sub2_weights

            elif FLAGS.attn in ['multihead', 'selfattn']:
                # 没动
                return self.sub1_attn_weight_list, self.sub2_attn_weight_list


    def inverse_attn_or_gradient(self, attn_or_gard):
        C = 1 if FLAGS.attn.startswith('sim') else self.num_classes
        E_num = self.num_nodes[0]
        # TODO: 多层的话，可以将attention weight们，先乘在一起，然后送到下面进行计算？   ----好像不能，因为是二跳邻居了
        col_idx = [tf.tile(tf.reshape(self.features[0], [E_num, 1, -1]), multiples=[1, C, 1])]  # (E, C, N_E)
        for t in range(self.num_node_types):
            col_idx_t = safe_lookup(self.features[t + 1], self.E4N[t], "sub1_col_idx_{}".format(t))  # (E, N_t, F_Nt)
            col_idx_t = tf.tile(tf.reshape(col_idx_t, [E_num, 1, -1]), multiples=[1, C, 1])  # (E, C, N_t*F_Nt)
            col_idx.append(col_idx_t)
        col_idx = tf.concat(col_idx, axis=-1)  # (E, C, F_E+\sum_t{N_t*F_Nt})
        row_idx = tf.reshape(tf.cast(tf.range(E_num), tf.int64), [-1, 1, 1])  # (E, 1, 1)
        row_idx = tf.tile(row_idx, multiples=[1, C, tf.shape(col_idx)[-1]])  # (E, C, F_E+\sum_t{N_t*F_Nt})
        chn_idx = tf.reshape(tf.cast(tf.range(C), tf.int64), [1, -1, 1])  # (1, C, 1)
        chn_idx = tf.tile(chn_idx, multiples=[E_num, 1, tf.shape(col_idx)[-1]])  # (E, C, F_E+\sum_t{N_t*F_Nt})
        indices = tf.reshape(tf.stack([row_idx, chn_idx, col_idx], axis=-1), [-1, 3])
        values = tf.reshape(attn_or_gard, [-1])  # (E * C * (F_E + \sum_t{N_t * F_Nt}))

        weights = tf.SparseTensor(indices=indices, values=values, dense_shape=[E_num, C, self.feature_vocab_size])
        weights = tf.sparse.to_dense(weights, validate_indices=False)

        return weights



if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from test_data import test_model_para
    from test_data import test_data

    # from Utils import DataConfig

    model = SceneMiningBaselineModel(**test_model_para)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run([model.attn_weights()])

    if hasattr(output, 'shape'):
        print(output.shape)
    else:
        for i in output:
            print(i)
            print(i.shape)
