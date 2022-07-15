import tensorflow as tf

def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
    return dims


def mask_score(scores, sequence_lengths, score_mask_value=tf.constant(-1e15, dtype=tf.float32)):
    score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
    score_mask_values = score_mask_value * tf.ones_like(scores)
    return tf.where(score_mask, scores, score_mask_values)


def mask_score_channel(scores, sequence_lengths, channel, score_mask_value=tf.constant(-1e15, dtype=tf.float32)):
    score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
    score_mask_values = score_mask_value * tf.ones_like(scores)
    return tf.where(tf.tile(score_mask, [channel, 1]), scores, score_mask_values)