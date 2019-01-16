import numpy as np
import tensorflow as tf
import os
import LoadData as LD
from tensorflow.core.framework import summary_pb2
import datetime
import deep_Core.green_functions as gf
from numba import cuda
import deep_Core.neural_net_functions as nn

def main():
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
    use_gdm = True
    learning_rate = 1e-4
    im_width = 28
    im_height = 28
    im_pix = im_height * im_width
    num_steps = 20000



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
        X = tf.placeholder(tf.float32, shape=(batch_size, im_width, im_height, 1))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, 10))

        # validation data
        tf_valX = tf.placeholder(tf.float32, shape=(len(valX), im_width, im_height, 1))

        # test data
        tf_testX = tf.placeholder(tf.float32, shape=(test_batch_size, im_width, im_height, 1))

        def createWeight(size, Name):
            return tf.Variable(tf.truncated_normal(size, stddev=0.1), name=Name)

        def createBias(size, Name):
            return tf.Variable(tf.constant(0.1, shape=size), name=Name)

        def conv2d_s1(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_3x3_s1(x):
            return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        def split_and_concat(array, num_split):
            arrays = [[] for _ in range(num_split)]
            for elem in array:
                elems = tf.split(elem, num_or_size_splits=num_split, axis=3)
                for idx, array in enumerate(arrays):
                    array.append(elems[idx])

            tensors = []
            for idx, array in enumerate(arrays):
                tensors.append(tf.concat(array, axis=3))

            return tensors

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
        if use_gdm:
            input_map1 = int(4 * output_map1)
            input_map2 = int(4 * output_map2)
        else:
            input_map1 = 4 * output_map1
            input_map2 = 4 * output_map2
        #
        # follows inception1
        W_conv2_1x1_1 = createWeight([1, 1, input_map1, output_map2], 'W_conv2_1x1_1')
        b_conv2_1x1_1 = createWeight([output_map2], 'b_conv2_1x1_1')

        # follows inception1
        W_conv2_1x1_2 = createWeight([1, 1, input_map1, OutputConv1x1], 'W_conv2_1x1_2')
        b_conv2_1x1_2 = createWeight([OutputConv1x1], 'b_conv2_1x1_2')

        # follows inception1
        W_conv2_1x1_3 = createWeight([1, 1, input_map1, OutputConv1x1], 'W_conv2_1x1_3')
        b_conv2_1x1_3 = createWeight([OutputConv1x1], 'b_conv2_1x1_3')

        # follows 1x1_2
        W_conv2_3x3 = createWeight([3, 3, OutputConv1x1, output_map2], 'W_conv2_3x3')
        b_conv2_3x3 = createWeight([output_map2], 'b_conv2_3x3')

        # follows 1x1_3
        W_conv2_5x5 = createWeight([5, 5, OutputConv1x1, output_map2], 'W_conv2_5x5')
        b_conv2_5x5 = createBias([output_map2], 'b_conv2_5x5')

        # follows max pooling
        W_conv2_1x1_4 = createWeight([1, 1, input_map1, output_map2], 'W_conv2_1x1_4')
        b_conv2_1x1_4 = createWeight([output_map2], 'b_conv2_1x1_4')

        ############ Fully connected layers #############
        # since padding is same, the feature map with there will be 4 28*28*output_map2
        W_fc1 = createWeight([input_map2 * im_pix, no_HiddenNodes], 'W_fc1')
        b_fc1 = createBias([no_HiddenNodes], 'b_fc1')

        W_fc2 = createWeight([no_HiddenNodes, no_OutputNodes], 'W_fc2')
        b_fc2 = createBias([no_OutputNodes], 'b_fc2')

        greens_function = gf.create_green_function((im_width, im_height))

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
            # inception1_temp = tf.nn.relu(tf.concat([conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4], axis=3))
            # inception1_I1, inception1_C1x, inception1_C1y = tf.split(inception1_temp, 3, axis=3)
            # inception1_gf = gf.gradient_domain_merging_with_orientation(inception1_I1, inception1_C1x, inception1_C1y, greens_function)
            # inception1_gf_1x1, _ = nn.conv_layer_uniform(inception1_gf, 1, num_output=int(input_map1/4),
            #                                          name='in1_gf_1x1', add_bias=True, add_relu=False)
            # inception1 = tf.concat([inception1_temp, inception1_gf_1x1], axis=3)
            # inception1 = inception1_temp

            inception1_C1x, inception1_C1y = split_and_concat([conv1_1x1_1, conv1_3x3, conv1_5x5, conv1_1x1_4],
                                                              num_split=2)

            inception1_gf = gf.gradient_domain_integration(inception1_C1x, inception1_C1y, greens_function)
            in1x, in1y = tf.image.image_gradients(inception1_gf)
            inception1 = tf.concat([tf.nn.relu(in1x), tf.nn.relu(in1y)], axis=3)

            # Inception Module 2
            conv2_1x1_1 = conv2d_s1(inception1, W_conv2_1x1_1) + b_conv2_1x1_1
            conv2_1x1_2 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_2) + b_conv2_1x1_2)
            conv2_1x1_3 = tf.nn.relu(conv2d_s1(inception1, W_conv2_1x1_3) + b_conv2_1x1_3)
            conv2_3x3 = conv2d_s1(conv2_1x1_2, W_conv2_3x3) + b_conv2_3x3
            conv2_5x5 = conv2d_s1(conv2_1x1_3, W_conv2_5x5) + b_conv2_5x5
            maxpool2 = max_pool_3x3_s1(inception1)
            conv2_1x1_4 = conv2d_s1(maxpool2, W_conv2_1x1_4) + b_conv2_1x1_4




            # concatenate all the feature maps and add a relu
            # inception2_temp = tf.nn.relu(tf.concat([conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4], axis=3))
            # inception2_temp_1x1, _ = nn.conv_layer_uniform(inception2_temp, 1, num_output=(4 * output_map2),
            #                                                name='in2_temp_1x1', add_bias=True, add_relu=False)

            inception2_C1x, inception2_C1y = split_and_concat([conv2_1x1_1, conv2_3x3, conv2_5x5, conv2_1x1_4], num_split=2)
            #
            # inception2_I1, inception2_C1x, inception2_C1y = tf.split(inception2_temp_1x1, 3, axis=3)
            # inception2_gf = gf.gradient_domain_merging_with_orientation(inception2_I1, inception2_C1x, inception2_C1y, greens_function)
            inception2_gf = gf.gradient_domain_integration(inception2_C1x, inception2_C1y, greens_function)
            in2x, in2y = tf.image.image_gradients(inception2_gf)
            # inception2 = tf.concat([inception2_I1, inception2_C1x, inception2_C1y, inception2_gf], axis=3)
            # inception2 = tf.concat([inception2_gf, in2x, in2y], axis=
            inception2 = tf.concat([tf.nn.relu(in2x), tf.nn.relu(in2y)], axis=3)

            # in2x, in2y = tf.split(inception2_temp, 2)
            # inception2 = gf.gradient_domain_integration(in2x, in2y, greens_function)

            # flatten features for fully connected layer
            inception2_flat = tf.reshape(inception2, [-1, im_pix * input_map2])

            # Fully connected layers
            if train:
                h_fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), dropout_rate)
            else:
                h_fc1 = tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1)

            return tf.matmul(h_fc1, W_fc2) + b_fc2

        this_model = model(X)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=this_model, labels=y_))
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        predictions_val = tf.nn.softmax(model(tf_valX, train=False))
        predictions_test = tf.nn.softmax(model(tf_testX, train=False))

        # initialize variable
        init = tf.initialize_all_variables()

        # use to save variables so we can pick up later
        saver = tf.train.Saver()

    sess = tf.Session(graph=graph)

    # initialize variables
    sess.run(init)
    print("Model initialized.")
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    val_writer_path = os.path.join('log_files', 'val_' + time_str)
    if not os.path.exists(val_writer_path):
        os.makedirs(val_writer_path)
    val_writer = tf.summary.FileWriter(val_writer_path)


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
            val_accuracy = accuracy(val_label, preds)
            this_summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag='val_accuracy', simple_value=val_accuracy)])
            val_writer.add_summary(this_summary, s)

            print("step: " + str(s))
            print("validation accuracy: " + str(val_accuracy))
            print(" ")

        # get test accuracy and save model
        if s == (num_steps - 1):
            # create an array to store the outputs for the test
            result = np.array([]).reshape(0, 10)

            for i in range(int(len(testX) / test_batch_size)):
                feed_dict = {tf_testX: data.next_test_batches(testX, test_batch_size)}
                preds = sess.run(predictions_test, feed_dict=feed_dict)
                result = np.concatenate((result, preds), axis=0)

            print("test accuracy: " + str(accuracy(test_label, result)))
            save_path = saver.save(sess, Model_file_path)
            print("Model saved.")

    sess.close()
    cuda.close()


if __name__ == "__main__":
    main()

