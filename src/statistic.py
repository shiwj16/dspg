import os
import csv
import tensorflow as tf

from logging import getLogger
logger = getLogger(__name__)

class Statistic(object):
    def __init__(self, sess, conf, model_dir, data_dir, max_to_keep=5):
        self.sess = sess
        self.env_name = conf.env_name
        self.save_interval = conf.eval_interval

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.writer = tf.summary.FileWriter(data_dir, sess.graph)
        self.placeholders = {}
        self.summaries = {}
        saved_variables = ['eval_epi_rewards', 'train_epi_rewards', 'Q_value', 'Q_loss', 'pi_loss']
        for variable in saved_variables:
            self.register_summary(variable, dtype=tf.float32)


    def register_summary(self, name, dtype):
        placeholder = tf.placeholder(dtype, [], name=name)
        self.placeholders[name] = placeholder
        self.summaries[name] = tf.summary.scalar(name + '_summary', placeholder)


    def add_summary(self, name, value, global_step):
        placeholder = self.placeholders[name]
        summary = self.summaries[name]
        out, _ = self.sess.run(
            [summary, placeholder],
            feed_dict={placeholder: value}
        )
        self.writer.add_summary(out, global_step)


    def save_step(self, global_step, epi_rewards, Q=None, Q_loss=None, pi_loss=None):
        # save epi_rewards and other variables
        if Q_loss is None:
            # evaluation step
            self.add_summary('eval_epi_rewards', epi_rewards, global_step)
            with open(self.data_dir + "/eval.csv", "a", newline='') as csv_file:
                csv.writer(csv_file).writerow([global_step, epi_rewards])
        else:
            # train step
            self.add_summary('train_epi_rewards', epi_rewards, global_step)
            self.add_summary('Q_value', Q, global_step)
            self.add_summary('Q_loss', Q_loss, global_step)
            self.add_summary('pi_loss', pi_loss, global_step)
            with open(self.data_dir + "/train.csv", "a", newline='') as csv_file:
                csv.writer(csv_file).writerow([global_step, epi_rewards, Q, Q_loss, pi_loss])
            # save models
            if global_step % self.save_interval == 0:
                logger.info("Saving checkpoints...")
                self.saver.save(self.sess, self.model_dir + '/model', global_step=global_step)


    def load_model(self):
        logger.info("Loading checkpoints...")
        pre_model_dir = os.path.dirname(self.model_dir) + '/model'

        ckpt = tf.train.get_checkpoint_state(pre_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(pre_model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            logger.info("Load SUCCESS: %s" % fname)
        else:
            logger.info("Load FAILED: %s" % pre_model_dir)

