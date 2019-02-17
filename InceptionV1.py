import numpy as np
import tensorflow as tf
import os
import LoadData as LD
from tensorflow.core.framework import summary_pb2
import datetime
import deep_Core.green_functions as gf
from numba import cuda
import deep_Core.inception_modules as im

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
    no_HiddenNodes = 700 #1024
    no_OutputNodes = 10
    dropout_rate=0.5
    use_gid = True
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
        X = tf.placeholder(tf.float32, shape=(None, im_width, im_height, 1))
        y_ = tf.placeholder(tf.float32, shape=(None, 10))

        def createWeight(size, Name):
            return tf.Variable(tf.truncated_normal(size, stddev=0.1), name=Name)

        def createBias(size, Name):
            return tf.Variable(tf.constant(0.1, shape=size), name=Name)


        input_map1 = 4 * output_map1
        input_map2 = 4 * output_map2


        ############ Fully connected layers #############
        # since padding is same, the feature map with there will be 4 28*28*output_map2
        W_fc1 = createWeight([input_map2 * im_pix, no_HiddenNodes], 'W_fc1')
        b_fc1 = createBias([no_HiddenNodes], 'b_fc1')

        W_fc2 = createWeight([no_HiddenNodes, no_OutputNodes], 'W_fc2')
        b_fc2 = createBias([no_OutputNodes], 'b_fc2')

        if use_gid:
            green_function = gf.create_green_function((im_width, im_height))
        else:
            green_function = None

        def model(x):

            in_1 = im.inception_module_v1(x, output_map1, 'in_1', green_function=green_function)
            in_2 = im.inception_module_v1(in_1, output_map2, 'in_2', green_function=green_function)

            # flatten features for fully connected layer
            inception2_flat = tf.reshape(in_2, [-1, im_pix * input_map2])

            # Fully connected layers
            h_fc1_train = tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1), dropout_rate)
            out_train = tf.matmul(h_fc1_train, W_fc2) + b_fc2

            h_fc1_not_train = tf.nn.relu(tf.matmul(inception2_flat, W_fc1) + b_fc1)
            out_not_train = tf.matmul(h_fc1_not_train, W_fc2) + b_fc2
            return out_train, out_not_train

        out_train, out_not_train = model(X)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_train, labels=y_))
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        predictions_val = tf.nn.softmax(out_not_train)
        predictions_test = tf.nn.softmax(out_not_train)

        # initialize variable
        init = tf.initialize_all_variables()

        # use to save variables so we can pick up later
        saver = tf.train.Saver()

    sess = tf.Session(graph=graph)

    # initialize variables
    run_metadata = tf.RunMetadata()
    sess.run(init, run_metadata=run_metadata)
    print("Model initialized.")
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    val_writer_path = os.path.join('log_files', 'val_' + time_str)
    if not os.path.exists(val_writer_path):
        os.makedirs(val_writer_path)
    val_writer = tf.summary.FileWriter(val_writer_path, sess.graph)
    val_writer.add_run_metadata(run_metadata, 'step%d' % 0)




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
            feed_dict = {X: valX}
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
                feed_dict = {X: data.next_test_batches(testX, test_batch_size)}
                preds = sess.run(predictions_test, feed_dict=feed_dict)
                result = np.concatenate((result, preds), axis=0)

            print("test accuracy: " + str(accuracy(test_label, result)))
            save_path = saver.save(sess, Model_file_path)
            print("Model saved.")

    sess.close()
    cuda.close()


if __name__ == "__main__":
    main()

