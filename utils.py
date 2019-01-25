import os
import gym
import numpy as np
from six import iteritems
from datetime import datetime

from logging import getLogger
logger = getLogger(__name__)


def action_converter(env, action):
    '''convert action from [-inf, inf] to true value [low, high].'''
    if type(env.action_space) is gym.spaces.Discrete:
            return np.argmax(action)
    else:
        action = np.tanh(action)   # from [-inf,inf] to [-1, 1]
        high, low = env.action_space.high, env.action_space.low
        return ((action + 1) / 2) * (high - low) + low


def get_dirs(config, items=None):
    attrs = config.flag_values_dict()
    # attrs = config.__dict__['__flags']
    keys = sorted(attrs)
    keys.remove('env_name')
    keys = ['env_name'] + keys

    names =[]
    for key in keys:
        # Only use useful flags
        if key in items:
            names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
                if type(attrs[key]) == list else attrs[key]))
    names.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_dir = os.path.expanduser((os.path.join('logs', *names) + '/models').replace('\\', '/'))
    data_dir = os.path.expanduser((os.path.join('logs', *names) + '/data').replace('\\', '/'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    logger.info('Model directory: %s' % model_dir)
    logger.info('Data directory: %s' % data_dir)
    return model_dir, data_dir


def preprocess_conf(conf, sub_path):
    options = conf.__flags

    for option, value in options.items():
        value = value.value
        if option == 'hidden_dims_pi':
            conf.hidden_dims_pi = eval(value)
        elif option == 'hidden_dims_V':
            conf.hidden_dims_V = eval(value)
        elif option == 'hidden_dims_Q':
            conf.hidden_dims_Q = eval(value)
        with open(os.path.dirname(sub_path) + '/Arguments.txt', 'w') as f:
            for key, value in iteritems(vars(conf)):
                f.write('%s: %s\n' % (key, str(value)))


def onehot(idx, num_entries):
    x = np.zeros(num_entries)
    x[idx] = 1
    return x


def horz_stack_images(*images, spacing=5, background_color=(0,0,0)):
    # assert that all shapes have the same siz
    if len(set([tuple(image.shape) for image in images])) != 1:
        raise Exception('All images must have same shape')
    if images[0].shape[2] != len(background_color):
        raise Exception('Depth of background color must be the same as depth of image.')
    height = images[0].shape[0]
    width = images[0].shape[1]
    depth = images[0].shape[2]
    canvas = np.ones([height, width*len(images) + spacing*(len(images) - 1), depth])
    bg_color = np.reshape(background_color, [1, 1, depth])
    canvas *= bg_color
    width_pos = 0
    for image in images:
        canvas[:, width_pos:width_pos+width, :] = image
        width_pos += (width + spacing)
    return canvas
