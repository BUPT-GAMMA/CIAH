import tensorflow as tf
from FeatureSelection.utils import get_shape, mask_score


transform_V = False

def simple_attention(inputs, masks, dropout, is_training, att_dim=50, scope=None):
    """
        simple attention to combine feature tensor of shape (B, L, D) into (B, D)
        and the corresponding weight tensor (B, L)
    :param inputs:  (B, L, D)
    :param att_dim:
    :param sequence_lengths:
    :param scope:
    :return:  [ (B, D) ,  (B, L, D) ],  (B, L)
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    D_w = get_shape(inputs)[-1]
    N_w = get_shape(inputs)[-2]

    with tf.variable_scope(scope or 'attention'):
        inputs = tf.layers.dropout(inputs, rate=dropout, training=tf.convert_to_tensor(is_training))
        # input_proj 实际上就是 QKV 中的 K
        W = tf.get_variable('attn_K_W', shape=[D_w, att_dim])
        b = tf.get_variable('attn_K_b', shape=[att_dim])
        K = tf.nn.tanh(tf.matmul(tf.reshape(inputs, [-1, D_w]), W) + b)
        # feature_att_W 实际上就是 QKV 中的 Q
        Q = tf.get_variable(name='attn_Q', shape=[att_dim, 1])
        alpha_score = tf.matmul(K, Q)

        alpha_score = tf.reshape(alpha_score, shape=[-1, N_w])
        alpha_score = alpha_score + tf.cast(~masks, tf.float32) * (-1e15)  # mask要取反，盖住 mask=False(即padding) 的部分
        alpha_weight = tf.nn.softmax(alpha_score, name="selected_weight")

        if transform_V:
            Wv = tf.get_variable('attn_V_W', shape=[D_w, D_w])
            bv = tf.get_variable('attn_V_b', shape=[D_w])
            V = tf.nn.tanh(tf.matmul(inputs, Wv) + bv)
        else:
            # 这里 V 就是原本的inputs
            V = inputs
        outputs = tf.reduce_sum(V * tf.expand_dims(alpha_weight, 2), axis=1,
                                name="selected_feature")

        return [outputs, V], alpha_weight        # (B, D),  (B, L)


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import numpy as np

    # inputs, sequence_lengths, att_dim, scope = None
    inputs = np.random.randint(0, 10, size=(5, 4, 3)) / 11
    lengths = np.array([0,0,2,1,0])

    inputs_placeholder = tf.placeholder(tf.float32, [5, 4, 3], name="input_x")
    sequence_lengths_placeholder = tf.placeholder(tf.int32, [5], name="input_x_lengths")
    att_dim = 2
    outputs, alpha = simple_attention(inputs_placeholder, sequence_lengths_placeholder, att_dim)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run([outputs, alpha], feed_dict={
            inputs_placeholder: inputs,
            sequence_lengths_placeholder: lengths
        })

    if hasattr(output, 'shape'):
        print(output.shape)
    else:
        for i in output:
            print(i)
            print(i.shape)
