import tensorflow as tf
import tensorflow.contrib.layers as layers

EPS = 1E-6

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)


def get_policy(pi_type, hidden_dims, reg, reg_factor):
    if pi_type == 'Gaussian':
        return GaussianPolicy(hidden_dims, reg, reg_factor)
    elif pi_type == 'MixtureGaussian':
        return GaussianMixturePolicy(hidden_dims, reg, reg_factor)
    elif pi_type == 'Categorical':
        return CategoricalPolicy(hidden_dims, reg, reg_factor)
    else:
        raise ValueError('Unknown policy type: %s' % pi_type)


class MLPPolicy(object):
    def __init__(self, hidden_dims, reg, reg_factor):
        self.hidden_dims = hidden_dims
        self.regularizer = layers.l2_regularizer(scale=reg_factor) if (reg is True) else None

    def feature_extractor(self, state):
        fc = state
        for i, hidden_dim in enumerate(self.hidden_dims):
            fc = tf.layers.dense(fc, hidden_dim, tf.nn.relu,
                                 kernel_regularizer=self.regularizer, name='fc{}'.format(i))
        return fc


class GaussianPolicy(MLPPolicy):
    '''
    Policy outputs a gaussian action that is clamped to the interval [-1, 1]
    '''
    def __init__(self, hidden_dims, reg, reg_factor):
        super(GaussianPolicy, self).__init__(hidden_dims, reg, reg_factor)

    def produce_policy_parameters(self, a_shape, processed_s):
        mu_params = tf.layers.dense(processed_s, a_shape,
                                    kernel_regularizer=self.regularizer, name='mu_params')
        sigma_params = tf.layers.dense(processed_s, a_shape, tf.nn.sigmoid,
                                       kernel_regularizer=self.regularizer, name='sigma_params')
        return (mu_params, sigma_params + 0.0001)

    def policy_parameters_to_log_prob(self, u, parameters):
        (mu, sigma) = parameters
        log_prob = tf.distributions.Normal(mu, sigma).log_prob(u)
        #print(log_prob)
        return tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(u)) + EPS), axis=1)

    def policy_parameters_to_sample(self, parameters):
        (mu, sigma) = parameters
        return tf.distributions.Normal(mu, sigma).sample()


class GaussianMixturePolicy(MLPPolicy):
    def __init__(self, hidden_dims, reg, reg_factor):
        super(GaussianMixturePolicy, self).__init__(hidden_dims, reg, reg_factor)

    def produce_policy_parameters(self, a_shape, processed_s):
        pass

    def policy_parmeters_to_log_prob(self, a, parameters):
        pass

    def policy_parameters_to_sample(self, parameters):
        pass


class CategoricalPolicy(MLPPolicy):
    def __init__(self, hidden_dims, reg, reg_factor):
        super(CategoricalPolicy, self).__init__(hidden_dims, reg, reg_factor)

    def produce_policy_parameters(self, a_shape, processed_s):
        logits = tf.layers.dense(processed_s, a_shape,
                                 kernel_regularizer=self.regularizer, name='logits')
        return logits

    def policy_parameters_to_log_prob(self, a, parameters):
        logits = parameters
        out = tf.distributions.Categorical(logits=logits).log_prob(tf.argmax(a, axis=1))
        #out = tf.Print(out, [out], summarize=10)
        return out

    def policy_parameters_to_sample(self, parameters):
        logits = parameters
        a_shape = logits.get_shape()[1].value
        #logits = tf.Print(logits, [tf.nn.softmax(logits)], message='logits are:', summarize=10)
        out = tf.one_hot(tf.distributions.Categorical(logits=logits).sample(), a_shape)
        return out