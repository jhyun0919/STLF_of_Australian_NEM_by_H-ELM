# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle as pickle
from data_parser import data_alloter

file_directory = '/Users/JH/Desktop/NTU/NTU_Research/data/NEM_Load_Forecasting_Database.xls'
logs_path = './tensorflow_logs/cnn_preprocessed'
test_result_directory = './cnn_preprocessed_result.bin'

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
n_input = 144  # feature data as input (input matrix shape: ???)
n_output = 48  # target data as output  (48-points)
dropout = 0.8  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name='feature_input')
y = tf.placeholder(tf.float32, [None, n_output], name='target_output')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout (keep probability)
learning_rate_decayed = tf.placeholder(tf.float32, shape=[], name='learning_rate_dacayed')


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    with tf.name_scope('array_reshape') as array_reshape:
        x = tf.reshape(x, shape=[-1, 12, 12, 1])

    # Convolution Layer
    with tf.name_scope('conv_layer1') as conv_layer1:
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    with tf.name_scope('conv_layer2') as conv_layer2:
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    with tf.name_scope('fc_layer') as fc_layer:
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    with tf.name_scope('output_layer') as output_layer:
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 outputs(number of filter = 32)
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 32]), name='wc1'),

    # 3x3 conv, 32 inputs, 64 outputs(number of filter = 64)
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 32, 64]), name='wc2'),

    # fully connected 1, width*height*64 inputs, ___ outputs
    'wd1': tf.Variable(tf.truncated_normal([3 * 3 * 64, 64 * 6]), name='wd1'),

    # ___ inputs, 48 outputs
    'out': tf.Variable(tf.truncated_normal([64 * 6, n_output]), name='wo1')
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.truncated_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.truncated_normal([64 * 6]), name='bd1'),
    'out': tf.Variable(tf.truncated_normal([n_output]), name='bo1')
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope('cost'):
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(tf.pow(y - pred, 2))

with tf.name_scope('optimization'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_decayed).minimize(cost)

# Evaluation model
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
#summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

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
        total_batch = int(data_set.PreProcessed.Train.feature.shape[0] / batch_size)

        # set a offset
        offset = (step * batch_size) % (data_set.PreProcessed.Train.target.shape[0] - batch_size)
        # Generate a minibatch
        batch_train_x = data_set.PreProcessed.Train.feature[offset:(offset + batch_size), :]
        batch_train_y = data_set.PreProcessed.Train.target[offset:(offset + batch_size), :]

        batch_test_x = data_set.PreProcessed.Test.feature
        batch_test_y = data_set.PreProcessed.Test.target

        # Run optimization op (backprop)
        _, c, summary_train = sess.run([optimizer, cost, merged_summary_op],
                                       feed_dict={x: batch_train_x, y: batch_train_y, keep_prob: dropout,
                                                  learning_rate_decayed: learning_rate})
        err_train = sess.run(rmse, feed_dict={x: batch_train_x, y: batch_train_y, keep_prob: 1.})

        err_test, summary_test = sess.run([rmse, merged_summary_op],
                                          feed_dict={x: batch_test_x, y: batch_test_y, keep_prob: 1.})

#summary_writer.add_summary(summary_train, step)
#summary_writer.add_summary(summary_test, step)

        # Compute average loss
        avg_cost += c / total_batch

        if step % data_showing_step == 0:
            print "step:", '%04d' % step, "cost=", "{:.9f}".format(avg_cost), \
                "rmse_train=", "{:.9f}".format(err_train), \
                "rmse_test=", "{:.9f}".format(err_test)

    print "Optimization Finished!"
    print

    print "Run the command line:\n" \
          "--> tensorboard --logdir=./tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser"

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

    n_simulations = 400

    test_result_recorder(data_set, n_simulations)
