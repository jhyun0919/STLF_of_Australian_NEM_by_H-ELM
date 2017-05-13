# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import cPickle as pickle
from data_parser import data_alloter

file_directory = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'
logs_path = './tensorflow_logs/lstm_raw'
test_result_directory = './rnn_raw_result.bin'

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'

# Parameters
num_steps = 3000
data_showing_step = 100
batch_size = 30

# Network Parameters
n_input = 10  # MNIST data input (img shape: 28*28)
n_steps = 5 # timesteps
n_hidden = 50  # hidden layer num of features
n_classes = 48  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='feature_input')
y = tf.placeholder(tf.float32, [None, n_classes], name='target_output')
learning_rate_decayed = tf.placeholder(tf.float32, shape=[], name='learning_rate_dacayed')

# Define weights
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')}
biases = {'out': tf.Variable(tf.random_normal([n_classes]), name='biases')}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    with tf.name_scope('array_reshape') as array_reshape:
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

    with tf.name_scope('lstm_cells') as lstm_cells:
        # Define a lstm cell with tensorflow
        lstm_cell_unit = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell_unit, x, dtype=tf.float32)

    with tf.name_scope('output_layer') as output_layer:
        # Linear activation, using rnn inner loop last output
        out = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

    return out


pred = RNN(x, weights, biases)

with tf.name_scope('cost'):
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y - pred, 2))

with tf.name_scope('optimization'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_decayed).minimize(cost)

with tf.name_scope('evaluation'):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor rmse tensor
tf.summary.scalar("RMSE", rmse)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


def run_graph(data_set):
    # Launch the graph
    sess = tf.InteractiveSession()
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for step in range(num_steps):

        # learning_rate_decayed
        if step < 1000:
            learning_rate = 0.001
        elif 1000 <= step < 1500:
            learning_rate = 0.00005
        else:
            learning_rate = 0.00000001

        avg_cost = 0.
        total_batch = int(data_set.Raw.Train.feature.shape[0] / batch_size)

        # set a offset
        offset = (step * batch_size) % (data_set.Raw.Train.target.shape[0] - batch_size)
        # Generate a minibatch.
        batch_train_x = data_set.Raw.Train.feature[offset:(offset + batch_size), :]
        batch_train_x = batch_train_x.reshape((batch_size, n_steps, n_input))
        batch_train_y = data_set.Raw.Train.target[offset:(offset + batch_size), :]

        test_len = len(data_set.Raw.Test.feature)
        batch_test_x = data_set.Raw.Test.feature[:test_len].reshape((-1, n_steps, n_input))
        batch_test_y = data_set.Raw.Test.target[:test_len]

        # Run optimization op (backprop)
        _, c, summary_train = sess.run([optimizer, cost, merged_summary_op],
                                       feed_dict={x: batch_train_x, y: batch_train_y,
                                                  learning_rate_decayed: learning_rate})
        err_train = sess.run(rmse, feed_dict={x: batch_train_x, y: batch_train_y})
        err_test, summary_test = sess.run([rmse, merged_summary_op], feed_dict={x: batch_test_x, y: batch_test_y})

        summary_writer.add_summary(summary_train, step)
        summary_writer.add_summary(summary_test, step)

        # Compute average loss
        avg_cost += c / total_batch

        if step % data_showing_step == 0:
            print "  step:", '%04d' % step, "cost=", "{:.9f}".format(avg_cost), \
                "rmse_train=", "{:.9f}".format(err_train), \
                "rmse_test=", "{:.9f}".format(err_test)

    print "  Optimization Finished!"
    print

    print "  Run the command line:\n" \
          "  --> tensorboard --logdir=./tensorflow_logs \n" \
          "  Then open http://0.0.0.0:6006/ into your web browser"

    return err_test


def test_result_recorder(data_set, n_simulatations):
    result = list()

    for i in xrange(0, n_simulatations):
        print '=> simulation number : ',
        print str(i)

        result.append(run_graph(data_set))

    f = open(test_result_directory, 'w')
    pickle.dump(np.array(result), f)
    f.close()


if __name__ == "__main__":
    df = pd.read_excel(file_directory, sheetname=QLD)
    data_set = data_alloter(df)

    n_simulations = 1000

    test_result_recorder(data_set, n_simulations)
