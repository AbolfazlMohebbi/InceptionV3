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
    output_map1 = 16
    output_map2 = 32
    no_HiddenNodes = 700 #1028
    no_OutputNodes = 10
    OutputConv1x1 = 16
    dropout_rate=0.5
    use_gid = False
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

        def gradient_integration_derivative(array_of_tensors, add_weights=True):
            # Split each array of the tensor and concat them as 2 tensors
            C1x_temp, C1y_temp = split_and_concat(array_of_tensors, num_split=2)

            # Add weights if necessary
            if add_weights:
                num_channels = tf.shape(C1x_temp)
                C1x, _ = nn.conv_layer_uniform(C1x_temp, 1, num_channels[3], 'weights_before_GDI_C1x', add_bias=False, add_relu=False)
                C1y, _ = nn.conv_layer_uniform(C1y_temp, 1, num_channels[3], 'weights_before_GDI_C1y', add_bias=False, add_relu=False)
            else:
                C1x = C1x_temp
                C1y = C1y_temp

            # Integrate then derivate the features
            integration = gf.gradient_domain_integration(C1x, C1y, greens_function)
            integration_dx, integration_dy = tf.image.image_gradients(integration)
            integration_gradient = tf.concat([tf.nn.relu(integration_dx), tf.nn.relu(integration_dy)], axis=3)
            return integration_gradient

        def inception_module(in_tensor, layers_output_channels, mod_name, use_gid=False):

            # Add all the convolutional layers
            num_out = layers_output_channels
            num_out_half = int(num_out/2)

            # 1st branch
            conv_1x1_1 = nn.conv_layer_truncated_normal(in_tensor, 1, num_out, name='conv_' + mod_name + '_1x1_1',
                                              add_relu=True, add_bias=True)

            # 2nd branch
            conv_1x1_2 = nn.conv_layer_truncated_normal(in_tensor, 1, num_out_half, name='conv_' + mod_name + '_1x1_2',
                                              add_relu=True, add_bias=True)
            conv_3x3_2 = nn.conv_layer_truncated_normal(conv_1x1_2, 3, num_out, name='conv_' + mod_name + '_3x3_2',
                                              add_relu=True, add_bias=True)

            # 3rd branch
            conv_1x1_3 = nn.conv_layer_truncated_normal(in_tensor, 1, num_out_half, name='conv_' + mod_name + '_1x1_3',
                                              add_relu=True, add_bias=True)
            conv_5x5_3 = nn.conv_layer_truncated_normal(conv_1x1_3, 5, num_out, name='conv_' + mod_name + '_5x5_3',
                                              add_relu=True, add_bias=True)

            # 4th branch
            maxpool_4 = max_pool_3x3_s1(in_tensor)
            conv_1x1_4 = nn.conv_layer_truncated_normal(maxpool_4, 1, num_out, name='conv_' + mod_name + '_1x1_4',
                                              add_relu=True, add_bias=True)

            # Use the gradient-integration-derivative if required, else concat the tensors
            tensor_array1 = [conv_1x1_1, conv_3x3_2, conv_5x5_3, conv_1x1_4]
            if use_gid:
                inception_out = gradient_integration_derivative(tensor_array1, add_weights=True)
            else:
                inception_out = tf.concat(tensor_array1, axis=3)

            return inception_out

        input_map1 = 4 * output_map1
        input_map2 = 4 * output_map2


        ############ Fully connected layers #############
        # since padding is same, the feature map with there will be 4 28*28*output_map2
        W_fc1 = createWeight([input_map2 * im_pix, no_HiddenNodes], 'W_fc1')
        b_fc1 = createBias([no_HiddenNodes], 'b_fc1')

        W_fc2 = createWeight([no_HiddenNodes, no_OutputNodes], 'W_fc2')
        b_fc2 = createBias([no_OutputNodes], 'b_fc2')

        greens_function = gf.create_green_function((im_width, im_height))

        def model(x, train=True):
            in_1 = inception_module(x, output_map1, 'in_1', use_gid=use_gid)
            in_2 = inception_module(in_1, output_map2, 'in_2', use_gid=use_gid)

            # flatten features for fully connected layer
            inception2_flat = tf.reshape(in_2, [-1, im_pix * input_map2])

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
            print("loss value: " + str(loss_value))
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

