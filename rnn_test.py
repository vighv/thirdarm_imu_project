import tensorflow as tf
import numpy as np
from constants import *
import matplotlib.pyplot as plt

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 35

class RNN(object):
    def __init__(self, X, y, config):
        self.config = config
        self.input_size = len(X)
        self.output_size = len(y)
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, config.num_steps, config.batch_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, config.num_steps, config.batch_size], name='ys')

        # Add hidden layer
        with tf.name_scope('hidden_in'):
            layer_input_x = tf.reshape(self.xs, [-1, self.input_size], name='to_2d')
            init = tf.random_normal_initializer(mean=0, stddev=1)
            w_in = tf.get_variable('weights', [self.input_size, config.hidden_size], initializer=init)
            init = tf.constant_initializer(0.1)
            b_in = tf.get_variable('bias', [config.hidden_size], initializer=init)
            with tf.name_scope('wx_plus_b'):
                layer_input_y = tf.matmul(layer_input_x, w_in) + b_in
                self.layer_y = tf.reshape(layer_input_y, [-1, config.num_steps, config.hidden_size], 'to_3d')

        # Add cells
        with tf.name_scope('lstm_cell'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, 1, True)
            with tf.name_scope('init_state'):
                self.init_cell_state = lstm_cell.zero_state(config.batch_size, tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.layer_y,
                initial_state=self.init_cell_state,
                time_major=False)

        # Add output layer
        with tf.name_scope('hidden_out'):
            layer_output_x = tf.reshape(self.cell_outputs, [-1, config.hidden_size], 'to_2d')
            w_out = tf.get_variable('weights', [self.output_size,], initializer=init)
            b_out = tf.get_variable('bias', [self.output_size,], initializer=init)
            with tf.name_scope('ws_plus_b'):
                self.pred = tf.matmul(layer_output_x, w_out) + b_out

        # Compute cost
        with tf.name_scope('cost'):
            losses = tf.contrib.seq2seq.sequence_loss(
                tf.reshape(self.pred, [-1], 'reshape_pred'),
                tf.reshape(self.ys, [-1], 'reshape_target'),
                tf.ones([config.batch_size * config.num_steps], dtype=tf.float32),
                average_across_timesteps=True)
            with tf.name_scope('average_cost'):
                self.cost = tf.div(tf.reduce_sum(losses, name='loss_sum'),
                                   tf.cast(config.batch_size, tf.float32),
                                   name='average_cost')
                tf.scalar_summary('cost', self.cost)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cost)


def init_rnn():
    pass


def train_rnn():
    config = MediumConfig()
    X = np.genfromtxt(X_FILE, delimiter=',')
    # X = X.reshape((config.batch_size, config.num_steps, :))
    y = np.genfromtxt(Y_FILE, delimiter=',')
    y = y.reshape((y.batch_size, y.num_steps))
    model = RNN(X, y, config)
    session = tf.Session()
    summary = tf.summary.merge_all()
    writer = tf.train.summary.FileWriter("logs", session.graph)
    session.run(tf.initialize_all_variables())



