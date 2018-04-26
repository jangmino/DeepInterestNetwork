import tensorflow as tf

def dice(_x, name='',training=False, trainable=False, reuse=False):
  with tf.variable_scope("dice", reuse=tf.AUTO_REUSE):
    alphas = tf.get_variable('alpha'+name, _x.get_shape()[-1],
                         initializer=tf.constant_initializer(0.0), trainable=trainable,
                         dtype=tf.float32)
  x_normed = tf.layers.batch_normalization(_x, name=name, center=False, scale=False, reuse=reuse, training=training, trainable=trainable) #a simple way to use BN to calculate x_p
  x_p = tf.sigmoid(x_normed)
  return alphas * (1.0 - x_p) * _x + x_p * _x
