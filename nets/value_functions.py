import tensorflow as tf


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

class MLPValueFunc(object):
    def __init__(self, hidden_dims_Q, hidden_dims_V):
        self.hidden_dims_Q = hidden_dims_Q
        self.hidden_dims_V = hidden_dims_V

    def Q_network(self, state, action, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc = tf.concat([state, action], axis=1)
            for i, hidden_dim in enumerate(self.hidden_dims_Q):
                fc = tf.layers.dense(fc, hidden_dim, tf.nn.relu, name='fc{}'.format(i))
            Q = tf.reshape(tf.layers.dense(fc, 1, name='last_Q'), [-1])
        return Q

    def V_network(self, state, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc = state
            for i, hidden_dim in enumerate(self.hidden_dims_V):
                fc = tf.layers.dense(fc, hidden_dim, tf.nn.relu, name='fc{}'.format(i))
            V = tf.reshape(tf.layers.dense(fc, 1, name='last_V'), [-1])
        return V


