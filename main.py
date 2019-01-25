# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils import get_dirs, preprocess_conf, action_converter
from src.replay_buffer import ReplayBuffer2
from src.statistic import Statistic
from src.model import SoftPolicyGradient

flags = tf.app.flags
# environment
flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of environment：\
                    Pendulum-v0、Acrobot-v1、MountainCarContinuous-v0')
flags.DEFINE_float('reward_scale', 1.0, 'The scale of reward')
flags.DEFINE_integer('random_seed', 123, 'random seed')

# Algorithm
flags.DEFINE_boolean('load_model', False, 'whether to load a previous model')

# network
flags.DEFINE_string('hidden_dims_pi', '[800, 800]', 'dimensions of hidden layers of actor')
flags.DEFINE_string('hidden_dims_Q', '[800, 800]', 'dimensions of hidden layers of critic(Q)')
flags.DEFINE_string('pi_type', 'Gaussian', 'The type of policy：Gaussian、Categorical、MixtureGaussian')
flags.DEFINE_boolean('reg', False, 'whether to add weight regularization')
flags.DEFINE_float('reg_factor', 1e-3, 'scale of weight regularization if reg is True')
flags.DEFINE_integer('num_samples', 100, 'Sample # times to approximate the expectation of current policy')

# replay buffer
flags.DEFINE_integer('buffer_size', 5 * 10 ** 6, 'size of replay buffer')
flags.DEFINE_integer('batch_size', 100, 'The size of batch for minibatch training')

# training
flags.DEFINE_float('tau', 1e-2, 'weight of soft target update')
flags.DEFINE_float('gamma', 0.99, 'discount factor of Q-learning')
flags.DEFINE_float('Q_critic_learning_rate', 5e-4, 'value of critic(Q) learning rate')
flags.DEFINE_float('actor_learning_rate', 5e-5, 'value of actor learning rate')
flags.DEFINE_float('global_norm', 3, 'global norm to clip the gradient for actor')
flags.DEFINE_integer('eval_interval', 10 ** 4, 'Evaluate the current policy every # steps')
flags.DEFINE_integer('eval_episodes', 7, 'Run # episodes in each evaluate step')
flags.DEFINE_integer('max_steps', 5 * 10 ** 6, 'maximum # of global steps to train')
flags.DEFINE_integer('num_train_steps', 4, 'The number of train steps in each time step')

# Debug
flags.DEFINE_boolean('display', False, 'display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# set random seed
# tf.set_random_seed(conf.random_seed)

def main(_):
    model_dir, data_dir = get_dirs(conf, ['env_name'])
    preprocess_conf(conf, model_dir)

    env = gym.make(conf.env_name)
    # env.seed(conf.random_seed)
    state_shape = env.observation_space.shape
    if type(env.action_space) is gym.spaces.Discrete:
        action_shape = env.action_space.n
    else:
        action_shape = env.action_space.shape[0]

    # replay buffer
    buffer = ReplayBuffer2(conf.buffer_size)

    # building agent
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # agent
        agent = SoftPolicyGradient(sess, conf, state_shape, action_shape)
        # statistic
        stat = Statistic(sess, conf, model_dir, data_dir)
        if conf.load_model:
            stat.load_model()

        episode, global_step, local_step = 0, 0, 0
        epi_rewards = 0
        total_Q, Q_loss, pi_loss = [], [], []
        state = env.reset()
        pbar = tqdm(total=conf.max_steps, dynamic_ncols=True)
        while global_step < conf.max_steps:
            # interaction with environment
            action = agent.sampling_actions([state], is_deterministic=False)[0] # [-inf, inf]
            next_state, reward, done, info = env.step(action_converter(env, action))
            global_step += 1
            local_step += 1
            epi_rewards += reward
            reward *= conf.reward_scale
            buffer.add_transition(state, action, reward, next_state, done)
            state = next_state

            # train step
            if buffer.size() >= conf.batch_size:
                for i in range(conf.num_train_steps):
                    transitions = buffer.get_transitions(conf.batch_size)
                    Q, single_Q_loss, single_pi_loss = agent.trainer(transitions)
                    total_Q.append(np.mean(Q))
                    Q_loss.append(single_Q_loss)
                    pi_loss.append(single_pi_loss)

            # evaluate step
            if global_step % conf.eval_interval == 0:
                ave_epi_rewards = np.mean(eval_step(env, agent))
                stat.save_step(global_step, ave_epi_rewards)
                print('\n[Evaluation] averaged_epi_rewards: %.3f' % ave_epi_rewards)

            if done:
                # save step
                stat.save_step(global_step, epi_rewards, np.mean(total_Q), np.mean(Q_loss), np.mean(pi_loss))
                pbar.update(local_step)
                pbar.set_description('Episode: %s, epi_rewards: %.3f, pi_loss: %.3f, Q_loss: %.3f' %
                       (episode+1, epi_rewards, np.mean(pi_loss), np.mean(Q_loss)))
                print()
                episode += 1
                local_step = 0
                epi_rewards = 0
                total_Q, Q_loss, pi_loss = [], [], []
                state = env.reset()
        pbar.close()


def eval_step(env, agent):
    n_epi_rewards = []
    for i in range(conf.eval_episodes):
        epi_rewards = 0
        local_step = 0
        state = env.reset()
        while True:
            local_step += 1
            action = agent.sampling_actions([state], is_deterministic=True)[0] # [-inf, inf]
            next_state, reward, done, info = env.step(action_converter(env, action))
            epi_rewards += reward
            state = next_state
            if done:
                break
        n_epi_rewards.append(epi_rewards)
    return n_epi_rewards


if __name__ == '__main__':
    tf.app.run()