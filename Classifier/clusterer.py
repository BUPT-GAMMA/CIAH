import tensorflow as tf
from Models import \
    SceneMiningBaselineModel
from Criterion import \
    emb_weight_dual_cluster_module, \
    node_belonging_to_hyperedge_NCEloss_module, \
    gradient_guided_attn_module

FLAGS = tf.flags.FLAGS
random_initial_for_center = True
center_transformation_for_weights = True
num_steps = 4

class ClusterBaseline(object):
    def __init__(self, embedding_model, input_data, embedding_size):
        self.global_step = tf.get_variable("global_step", shape=(),
                                           initializer=tf.zeros_initializer(), trainable=False)
        self.v = FLAGS.stu_v
        with tf.variable_scope("embedding_model"):
            if embedding_model.lower() == 'Basemodel'.lower():
                self.model = SceneMiningBaselineModel(input_data, embedding_size)
            else:
                raise KeyError("Unknown embedding_model name: {}".format(embedding_model))

            if FLAGS.coef_belong > 0:
                self.behavior_embedding, self.entity_embeddings = self.model.logits(entity=True)
            else:
                self.behavior_embedding = self.model.logits()
            self.N2E_attns, self.E2N_attns = self.model.attn_weights()

            if FLAGS.layer_aggr == 'concat':
                self.emb_size = self.model.layer_num * self.model.embedding_size
            else:
                self.emb_size = self.model.embedding_size
            self.dropout = self.model.dropout
            self.training = self.model.training

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

        self.losses = dict()

        self.lossL2_emb = tf.sqrt(1e-5 + tf.nn.l2_loss(self.model.feature_embedding) / self.model.num_nodes[0]) * max(FLAGS.coef_l2_emb, 0)
        self.losses.update({"l2_emb": self.lossL2_emb})

        self.integrated_gradient = tf.zeros_like(self.model.ori_attn)  # 代码兼容性

        if FLAGS.coef_grad > 0:
            self.models_grad, self.behavior_embeddings_grad, self.N2E_attns_grad, self.E2N_attns_grad = [], [], [], []
            with tf.variable_scope("embedding_model", reuse=True):
                for i in range(num_steps):
                    if embedding_model.lower() == 'Basemodel'.lower():
                        grad_model = SceneMiningBaselineModel(input_data, embedding_size, grad_alpha=i/num_steps, silent_print=True)
                    else:
                        raise NotImplementedError

                    self.models_grad.append(grad_model)
                    self.behavior_embeddings_grad.append(grad_model.logits())
                    weights = grad_model.attn_weights()
                    self.N2E_attns_grad.append(weights[0])
                    self.E2N_attns_grad.append(weights[1])


        if FLAGS.coef_dualcluster > 0:
            with tf.variable_scope("dual_cluster"):
                target_layer = self.model.target_layer    # 第一层
                C_num = self.model.num_classes
                D_wgt = self.N2E_attns[target_layer].shape[-1]
                D_emb = self.behavior_embedding.shape[-1]
                one_shot_for_each_type = tf.gather(
                    self.behavior_embedding,
                    tf.random_uniform((C_num,), minval=1, maxval=(self.model.num_nodes[0] // C_num - 1), dtype=tf.int32) + \
                        tf.range(0, C_num, dtype=tf.int32) * (self.model.num_nodes[0] // C_num)
                )
                if random_initial_for_center:
                    self.cluster_emb_center = tf.get_variable("cluster_emb_center", shape=(C_num, D_emb))
                else:
                    self.cluster_emb_center = tf.get_variable("cluster_emb_center", initializer=one_shot_for_each_type)

                if center_transformation_for_weights:
                    center_transformation = tf.layers.Dense(D_wgt, use_bias=True, name="center_transformation")
                    self.cluster_wgt_center = center_transformation(self.cluster_emb_center)
                else:
                    self.cluster_wgt_center = tf.get_variable("cluster_wgt_center", shape=(C_num, D_wgt))

                cluster_output = emb_weight_dual_cluster_module(self.model.num_nodes[0], C_num,
                                                                self.behavior_embedding, self.cluster_emb_center,
                                                                self.N2E_attns[target_layer], self.cluster_wgt_center,
                                                                self.global_step, self.dropout, self.training, self.v,
                                                                return_inertia=True)
                self.cluster_loss, self.cluster_prob, self.classification_predict, self.inertia = cluster_output
                self.cluster_predict = tf.argmax(self.classification_predict, axis=1)

                self.losses.update({"dual_cluster": FLAGS.coef_dualcluster * self.cluster_loss})


        if FLAGS.coef_belong > 0:
            with tf.variable_scope("belonging"):
                self.belonging_list = self.model.E4N
                self.belong_loss, (self.belong_pos_loss, self.belong_neg_loss)= node_belonging_to_hyperedge_NCEloss_module(
                    self.behavior_embedding, self.entity_embeddings,
                    self.belonging_list, self.model.num_nodes,
                    dropout=self.dropout, training=self.training)
                self.losses.update({"belonging": FLAGS.coef_belong * self.belong_loss})
            self.pretrain_op_bel, self.pretrain_op_bel_g = self.training_op(self.belong_loss + self.lossL2_emb)


        if FLAGS.coef_dualcluster > 0 and FLAGS.coef_grad > 0:
            self.grads_by_element = []
            for i in range(num_steps):
                with tf.variable_scope("dual_cluster", reuse=True):
                    cluster_output_grad = emb_weight_dual_cluster_module(self.model.num_nodes[0], C_num,
                                                                    self.behavior_embeddings_grad[i], self.cluster_emb_center,
                                                                    self.N2E_attns_grad[i][target_layer], self.cluster_wgt_center,
                                                                    self.global_step, self.dropout, self.training, self.v)
                    preds = tf.reduce_max(cluster_output_grad[2], axis=-1)
                    first_attn_inputs = self.models_grad[i].first_attn_inputs
                    self.grads_by_element.append(tf.gradients(ys=preds, xs=first_attn_inputs))

            with tf.variable_scope("attn_grad"):
                self.attn_grad_loss, integrated_gradient_compressed = gradient_guided_attn_module(
                    self.grads_by_element, num_steps,
                    self.model.sub1_attn_weight_list[target_layer])
                self.losses.update({"attn_grad": FLAGS.coef_grad * self.attn_grad_loss})
                self.integrated_gradient = self.model.inverse_attn_or_gradient(integrated_gradient_compressed)


        if FLAGS.coef_l2_net > 0:
            vars = tf.trainable_variables()
            self.lossL2_net = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * FLAGS.coef_l2_net
            self.losses.update({"l2_net": self.lossL2_net})


        self.loss = tf.add_n(list(self.losses.values()))
        self.train_op, self.grads_and_vars = self.training_op(self.loss)
        self.merged_summary = self.summary()

        # metrics
        log_reshape = tf.reshape(self.cluster_predict, [-1, ])
        lab_reshape = tf.reshape(self.model.labels, [-1, ])
        # msk_reshape = tf.reshape(self.model.mask_y, [-1])
        whole_mask = self.model.train_mask | self.model.val_mask | self.model.test_mask

        # msk_reshape = tf.reshape(self.model.test_mask, [-1])
        msk_reshape = tf.reshape(whole_mask, [-1])
        self.whole_mask = msk_reshape
        self.out_pred = log_reshape[msk_reshape]   # 移除padding
        self.out_labe = lab_reshape[msk_reshape]

        self.monitored_tensors = self.monitor_tensors()


    def training_op(self, loss):
        # train_op = opt.minimize(loss)
        # return train_op
        with tf.name_scope('train_optimizer'):
            var_list = tf.trainable_variables()
            grads_and_vars = self.optimizer.compute_gradients(loss, var_list)
            capped_grads_and_vars = [(tf.clip_by_value(g, -5., 5.), v) for g, v in grads_and_vars if g is not None]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = self.optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)

            gradient_output = [(v.name, (g if g is not None else tf.zeros(1)), v) for g, v in grads_and_vars]
        return apply_gradient_op, gradient_output


    def summary(self):
        for vname, g, v in self.grads_and_vars:
            tf.summary.histogram(v.name, v)
            tf.summary.histogram(v.name + "/grad", g)

        tf.summary.scalar("loss_all", self.loss)
        for loss_name, sub_loss in self.losses.items():
            tf.summary.scalar(loss_name, sub_loss)
        merged = tf.summary.merge_all()

        return merged

    def monitor_tensors(self):
        def monitor(tensor):
            return tf.nn.moments(tensor, axes=None)

        return monitor(self.behavior_embedding), monitor(self.model.feature_embedding), monitor(self.model.hyperedge_init_embedding)