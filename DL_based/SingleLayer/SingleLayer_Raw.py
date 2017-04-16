# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle as pickle
from data_parser import data_alloter

file_directory = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'
logs_path = './tensorflow_logs/single'
test_result_directory = './single_raw_result.bin'

QLD = 'Actual_Data_QLD'
NSW = 'Actual_Data_NSW'
VIC = 'Actual_Data_VIC'
SA = 'Actual_Data_SA'
TAS = 'Actual_Data_TAS'

CONSTANT = tf.app.flags

# Parameters
CONSTANT.DEFINE_integer('num_steps', 3000, 'number of iterations')
CONSTANT.DEFINE_integer('data_showing_step', 100, 'data showing step')
CONSTANT.DEFINE_integer('batch_size', 30, 'size of the batch')

# Network Parameter
CONSTANT.DEFINE_integer('n_hidden', 70, 'number of perceptron in a layer')
CONSTANT.DEFINE_integer('n_input', 50, 'number of input')
CONSTANT.DEFINE_integer('n_classes', 48, 'number of output')

# tf Graph input
x = tf.placeholder(tf.float32, [None, tf.app.flags.FLAGS.n_input], name='feature_input')
y = tf.placeholder(tf.float32, [None, tf.app.flags.FLAGS.n_classes], name='target_output')
learning_rate_decayed = tf.placeholder(tf.float32, shape=[], name='learning_rate_dacayed')


# Create model
def singlelayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope('hidden_layer1') as hidden_layer1:
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    with tf.name_scope('output_layer') as output_layer:
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([tf.app.flags.FLAGS.n_input, tf.app.flags.FLAGS.n_hidden]), name='hw1'),
    'out': tf.Variable(tf.random_normal([tf.app.flags.FLAGS.n_hidden, tf.app.flags.FLAGS.n_classes]), name='ow')
}
biases = {
    'b1': tf.Variable(tf.random_normal([tf.app.flags.FLAGS.n_hidden]), name='hd1'),
    'out': tf.Variable(tf.random_normal([tf.app.flags.FLAGS.n_classes]), name='ob')
}

pred = singlelayer_perceptron(x, weights, biases)

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
    for step in range(tf.app.flags.FLAGS.num_steps):

        # learning_rate_decayed
        if step < 1000:
            learning_rate = 0.001
        elif 1000 <= step < 1500:
            learning_rate = 0.00005
        else:
            learning_rate = 0.00000001

        avg_cost = 0.
        total_batch = int(data_set.Raw.Train.feature.shape[0] / tf.app.flags.FLAGS.batch_size)

        # set a offset
        offset = (step * tf.app.flags.FLAGS.batch_size) % (
            data_set.Raw.Train.target.shape[0] - tf.app.flags.FLAGS.batch_size)
        # Generate a minibatch.
        batch_train_x = data_set.Raw.Train.feature[offset:(offset + tf.app.flags.FLAGS.batch_size), :]
        batch_train_y = data_set.Raw.Train.target[offset:(offset + tf.app.flags.FLAGS.batch_size), :]

        batch_test_x = data_set.Raw.Test.feature
        batch_test_y = data_set.Raw.Test.target

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

        if step % tf.app.flags.FLAGS.data_showing_step == 0:
            print "step:", '%04d' % step, "cost=", "{:.9f}".format(avg_cost), \
                "rmse_train=", "{:.9f}".format(err_train), \
                "rmse_test=", "{:.9f}".format(err_test)

    print "Optimization Finished!"
    print

    print "Run the command line:\n" \
          "--> tensorboard --logdir=./tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser"


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

    n_simulations = 3

    test_result_recorder(data_set, n_simulations)
