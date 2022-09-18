import tensorflow as tf
from GNN import SubLayer_E2N, SubLayer_N2E
from Utils import safe_lookup

FLAGS = tf.flags.FLAGS

class BaseModel(object):
    def __init__(self,
                 input_data,
                 # num_nodes,
                 # num_features,
                 # num_classes,
                 # feature_vocab_size,
                 embedding_size,
                 silent_print=False,
                 # fs_config,
                 # gnn_config,
                 ):
        # self.num_nodes = num_nodes
        # self.num_features = num_features
        # self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.silent_print = silent_print
        # self.fs_config = fs_config
        # self.gnn_config = gnn_config

        self.dropout = FLAGS.dropout
        self.training = tf.placeholder_with_default(tf.convert_to_tensor(False), shape=(), name="training")

        self.layer_num = FLAGS.layer
        self.init_input(input_data)
        self.init_embedding_layer()
        # self.init_hypergraph_neural_network()


    def init_input(self, input_data):
        (E4N, N4E), features, feature_types, num_features, labels, num_classes, \
            train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = input_data

        def get_variable_with_default(nparray, dtype, name):
            return  tf.get_variable(
                name=name,
                shape=nparray.shape,
                dtype=dtype,
                initializer=tf.constant_initializer(nparray),
                trainable=False)
        # create_tensor = get_variable_with_default
        create_tensor = tf.convert_to_tensor

        # 后续可以考虑增加batch，目前是单图
        with tf.name_scope('input'):
            # adj
            if FLAGS.neighbor > 0:
                E4N = [e4n[:, :FLAGS.neighbor] for e4n in E4N]
                N4E = [n4e[:, :FLAGS.neighbor] for n4e in N4E]

            self.E4N = [ create_tensor(e4n, dtype=tf.int64, name="E4N_{}".format(i)) for i, e4n in enumerate(E4N) ]
            self.N4E = [ create_tensor(n4e, dtype=tf.int64, name="N4E_{}".format(i)) for i, n4e in enumerate(N4E) ]

            # feature
            self.features = [ create_tensor(f, dtype=tf.int64, name=feature_types[i]) for i, f in enumerate(features) ]
            self.mask_feats = [tf.not_equal(f, 0) for f in self.features]
            self.num_node_types = len(features) - 1   # 第一个是超边的
            self.num_nodes = [f.shape[0] for f in features]
            self.feature_types = feature_types
            self.feature_vocab_size = num_features

            # label
            self.labels = create_tensor(labels, dtype=tf.int64, name="labels")
            self.num_classes = num_classes

            # dataset splits
            self.train_idx = create_tensor(train_idx, dtype=tf.int64, name="train_idx")
            self.val_idx = create_tensor(val_idx, dtype=tf.int64, name="val_idx")
            self.test_idx = create_tensor(test_idx, dtype=tf.int64, name="test_idx")
            self.train_mask = tf.cast(create_tensor(train_mask, dtype=tf.bool, name="train_mask"), tf.bool)
            self.val_mask = tf.cast(create_tensor(val_mask, dtype=tf.bool, name="val_mask"), tf.bool)
            self.test_mask = tf.cast(create_tensor(test_mask, dtype=tf.bool, name="test_mask"), tf.bool)

            if not self.silent_print:
                print("Shape of E4N: ", " ".join([str(e.shape) for e in self.E4N]))
                print("Shape of N4E: ", " ".join([str(e.shape) for e in self.N4E]))
                print("Shape of features: ", " ".join([str(f.shape) for f in self.features]))
                print("Type of features: ", self.feature_types)
                print("Num of features: ", self.feature_vocab_size)
                print("Num of classes: ", self.num_classes)


    def init_embedding_layer(self):
        # 这个应该只对非behavior节点进行embedding
        with tf.variable_scope('embedding'):
            self.feature_embedding = tf.get_variable("feature_embedding",     # feature_vocab_size * D
                                                     shape=[self.feature_vocab_size, self.embedding_size],
                                                     initializer=tf.truncated_normal_initializer(),)
            # (N_n, N_f) = > (N_n, N_f, D)
            self.hyperedge_init_embedding = safe_lookup(
                self.feature_embedding, self.features[0], name="e_init_embedding"
            )
            # 注意 第一行是padding，所以得到的embedding每行都相同，是正常的现象。
            self.nodes_init_embedding = [
                safe_lookup(self.feature_embedding, f,
                    name="{}_init_embedding".format(self.feature_types[i+1].split("_")[0]))
                for i, f in enumerate(self.features[1: ])
            ]



