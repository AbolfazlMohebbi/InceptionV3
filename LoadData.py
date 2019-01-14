import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


class LoadData():

    def __init__(self, TrainPath, TestPath):
        self.TrainPath = TrainPath
        self.TestPath = TestPath
        self.batch_index = 0
        self.test_batch_size = 100
        return

    def LoadMNIST(self):
        train_set = pd.read_csv(self.TrainPath, header=None)
        test_set = pd.read_csv(self.TestPath, header=None)

        test_set.head()

        train_label = np.array(train_set[0])
        test_label = np.array(test_set[0])

        train_label = (np.arange(10) == train_label[:, None]).astype(np.float32)
        test_label = (np.arange(10) == test_label[:, None]).astype(np.float32)

        trainX = train_set.drop(0, axis=1)
        testX = test_set.drop(0, axis=1)

        trainX = np.array(trainX).astype(np.float32)
        testX = np.array(testX).astype(np.float32)

        # reformat the data
        trainX = trainX.reshape(len(trainX), 28, 28, 1)
        testX = testX.reshape(len(testX), 28, 28, 1)

        # get a validation set and remove it from the train set
        trainX, valX, train_label, val_label = trainX[0:(len(trainX) - 500), :, :, :], trainX[(len(trainX) - 500):len(trainX), :, :, :], \
                                         train_label[0:(len(trainX) - 500), :], train_label[(len(trainX) - 500):len(trainX),:]

        # make sure the images are alright
        plt.imshow(trainX.reshape(len(trainX), 28, 28)[0], cmap="Greys")
        plt.show()
        return trainX, testX, valX, train_label, test_label, val_label

    def next_test_batches(self, data, batch_size):
        if (batch_size + self.batch_index) > data.shape[0]:
            print("problem with batch size")
        batch = data[self.batch_index:(self.batch_index + batch_size), :, :, :]
        self.batch_index = self.batch_index + batch_size
        return batch

