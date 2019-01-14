import numpy as np
import tensorflow as tf
import os
import LoadData as LD

# data path
TrainPath = 'data\mnist_train.csv'
TestPath = 'data\mnist_test.csv'

# To load from a previous model
Model_file_path = os.getcwd()+'/model.ckpt'

#accuracy of model
def accuracy(target,predictions):
    return(100.0*np.sum(np.argmax(target,1) == np.argmax(predictions,1))/target.shape[0])

batch_size = 50
test_batch_size = 100
output_map1 = 32
output_map2 = 64
no_HiddenNodes = 700 #1028
no_OutputNodes = 10
OutputConv1x1 = 16
dropout_rate=0.5

# batch_size: training batch size
# output_map1: number of feature maps output by each tower inside the first Inception module
# output_map2: number of feature maps output by each tower inside the second Inception module
# no_HiddenNodes: number of hidden nodes
# No_OutputNodes: number of output nodes
# OutputConv1x1: number of feature maps output by each 1×1 convolution that precedes a large convolution
# dropout_rate: dropout rate for nodes in the hidden layer during training

# Load the data
data = LD.LoadData(TrainPath, TestPath)
trainX, testX, valX, train_label, test_label, val_label = data.LoadMNIST()

graph = tf.Graph()
with graph.as_default():
    # train data and labels
    X = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    y_ = tf.placeholder(tf.float32, shape=(batch_size, 10))

    # validation data
    tf_valX = tf.placeholder(tf.float32, shape=(len(valX), 28, 28, 1))

    # test data
    tf_testX = tf.placeholder(tf.float32, shape=(test_batch_size, 28, 28, 1))

    def createWeight(size, Name):
        return tf.Variable(tf.truncated_normal(size, stddev=0.1), name=Name)

    def createBias(size, Name):
        return tf.Variable(tf.constant(0.1, shape=size), name=Name)

    def conv2d_s1(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3x3_s1(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    ########### Inception Module 1 #############
    #
    # follows input
    W_conv1_1x1_1 = createWeight([1, 1, 1, output_map1], 'W_conv1_1x1_1')
    b_conv1_1x1_1 = createWeight([output_map1], 'b_conv1_1x1_1')

    # follows input
    W_conv1_1x1_2 = createWeight([1, 1, 1, OutputConv1x1], 'W_conv1_1x1_2')
    b_conv1_1x1_2 = createWeight([OutputConv1x1], 'b_conv1_1x1_2')

    # follows input
    W_conv1_1x1_3 = createWeight([1, 1, 1, OutputConv1x1], 'W_conv1_1x1_3')
    b_conv1_1x1_3 = createWeight([OutputConv1x1], 'b_conv1_1x1_3')

    # follows 1x1_2
    W_conv1_3x3 = createWeight([3, 3, OutputConv1x1, output_map1], 'W_conv1_3x3')
    b_conv1_3x3 = createWeight([output_map1], 'b_conv1_3x3')

    # follows 1x1_3
    W_conv1_5x5 = createWeight([5, 5, OutputConv1x1, output_map1], 'W_conv1_5x5')
    b_conv1_5x5 = createBias([output_map1], 'b_conv1_5x5')

    # follows max pooling
    W_conv1_1x1_4 = createWeight([1, 1, 1, output_map1], 'W_conv1_1x1_4')
    b_conv1_1x1_4 = createWeight([output_map1], 'b_conv1_1x1_4')

    ########### Inception Module 2 #############
    #
    # follows inception1
    W_conv2_1x1_1 = createWeight([1, 1, 4 * output_map1, output_map2], 'W_conv2_1x1_1')
    b_conv2_1x1_1 = createWeight([output_map2], 'b_conv2_1x1_1')

    # follows inception1
    W_conv2_1x1_2 = createWeight([1, 1, 4 * output_map1, OutputConv1x1], 'W_conv2_1x1_2')
    b_conv2_1x1_2 = createWeight([OutputConv1x1], 'b_conv2_1x1_2')

    # follows inception1
    W_conv2_1x1_3 = createWeight([1, 1, 4 * output_map1, OutputConv1x1], 'W_conv2_1x1_3')
    b_conv2_1x1_3 = createWeight([OutputConv1x1], 'b_conv2_1x1_3')

    # follows 1x1_2
    W_conv2_3x3 = createWeight([3, 3, OutputConv1x1, output_map2], 'W_conv2_3x3')
    b_conv2_3x3 = createWeight([output_map2], 'b_conv2_3x3')

    # follows 1x1_3
    W_conv2_5x5 = createWeight([5, 5, OutputConv1x1, output_map2], 'W_conv2_5x5')
    b_conv2_5x5 = createBias([output_map2], 'b_conv2_5x5')

    # follows max pooling
    W_conv2_1x1_4 = createWeight([1, 1, 4 * output_map1, output_map2], 'W_conv2_1x1_4')
    b_conv2_1x1_4 = createWeight([output_map2], 'b_conv2_1x1_4')

    ############ Fully connected layers #############
    # since padding is same, the feature map with there will be 4 28*28*output_map2
    W_fc1 = createWeight([28 * 28 * (4 * output_map2), no_HiddenNodes], 'W_fc1')
    b_fc1 = createBias([no_HiddenNodes], 'b_fc1')

    W_fc2 = createWeight([no_HiddenNodes, no_OutputNodes], 'W_fc2')
    b_fc2 = createBias([no_OutputNodes], 'b_fc2')

    def model(x, train=True):
        # Inception Module 1
        conv1_1x1_1 = conv2d_s1(x, W_conv1_1x1_1) + b_conv1_1x1_1
        conv1_1x1_2 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_2) + b_conv1_1x1_2)
        conv1_1x1_3 = tf.nn.relu(conv2d_s1(x, W_conv1_1x1_3) + b_conv1_1x1_3)
        conv1_3x3 = conv2d_s1(conv1_1x1_2, W_conv1_3x3) + b_conv1_3x3
        conv1_5x5 = conv2d_s1(conv1_1x1_3, W_conv1_5x5) + b_conv1_5x5
        maxpool1 = max_pool_3x3_s1(x)
        conv1_1x1_4 = conv2d_s1(maxpool1, W_conv1_1x1_4) + b_conv1_1x1_4

        # concatenate all the feature maps and add a relu
        inception1 = tf.nn.relu(tf.concat([conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4], axis=3))

        # Inception Module 2
        conv2_1x1_1 = conv2d_s1(inception1, W_conv2_1x1_1) + b_conv2_1x1_1
        conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_2) + b_conv2_1x1_2)
        conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_3) + b_conv2_1x1_3)
        conv2_3x3 = conv2d_s1(conv2_1x1_2, W_conv2_3x3) + b_conv2_3x3
        conv2_5x5 = conv2d_s1(conv2_1x1_3, W_conv2_5x5) + b_conv2_5x5
        maxpool2 = max_pool_3x3_s1(inception1)
        conv2_1x1_4 = conv2d_s1(maxpool2, W_conv2_1x1_4) + b_conv2_1x1_4

        # concatenate all the feature maps and add a relu
        inception2 = tf.nn.relu(tf.concat([conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4], axis=3))

        # flatten features for fully connected layer
        inception2_flat = tf.reshape(inception2, [-1, 28 * 28 * 4 * output_map2])

        # Fully connected layers
        if train:
            h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), dropout_rate)
        else:
            h_fc1 = tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1)

        return tf.matmul(h_fc1, W_fc2) + b_fc2


    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model(X), labels=y_))
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

    predictions_val = tf.nn.softmax(model(tf_valX, train=False))
    predictions_test = tf.nn.softmax(model(tf_testX, train=False))

    # initialize variable
    init = tf.initialize_all_variables()

    # use to save variables so we can pick up later
    saver = tf.train.Saver()

num_steps = 20000
sess = tf.Session(graph=graph)

# initialize variables
sess.run(init)
print("Model initialized.")

# set use_previous=1 to use file_path model
# set use_previous=0 to start model from scratch
use_previous = 0

# use the previous model or don't and initialize variables
if use_previous:
    saver.restore(sess, Model_file_path)
    print("Model restored.")

# training
for s in range(num_steps):
    offset = (s * batch_size) % (len(trainX) - batch_size)
    batch_x, batch_y = trainX[offset:(offset + batch_size), :], train_label[offset:(offset + batch_size), :]
    feed_dict = {X: batch_x, y_: batch_y}
    _, loss_value = sess.run([opt, loss], feed_dict=feed_dict)
    if s % 100 == 0:
        feed_dict = {tf_valX: valX}
        preds = sess.run(predictions_val, feed_dict=feed_dict)

        print("step: " + str(s))
        print("validation accuracy: " + str(accuracy(val_label, preds)))
        print(" ")

    # get test accuracy and save model
    if s == (num_steps - 1):
        # create an array to store the outputs for the test
        result = np.array([]).reshape(0, 10)

        for i in range(len(testX) / test_batch_size):
            feed_dict = {tf_testX: data.next_test_batches(testX, test_batch_size)}
            preds = sess.run(predictions_test, feed_dict=feed_dict)
            result = np.concatenate((result, preds), axis=0)

        print("test accuracy: " + str(accuracy(test_label, result)))
        save_path = saver.save(sess, Model_file_path)
        print("Model saved.")