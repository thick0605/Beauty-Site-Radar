import os
import sys
from tensorflow.contrib.learn import ModeKeys
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, training, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            print(path)
            vgg16_npy_path = path

        self.train = training
        self.vgg_weight = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)


    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def get_conv_filter(self, name):
        return tf.Variable(self.vgg_weight[name][0], name="filter")


    def get_bias(self, name):
        return tf.Variable(self.vgg_weight[name][1], name="biases")        
    
    def feature_extract(self, data_X, training, reuse=False):
        # Convert RGB to BGR
        red = data_X[:,:,:,0][:,:,:,np.newaxis]
        green = data_X[:,:,:,1][:,:,:,np.newaxis]
        blue = data_X[:,:,:,2][:,:,:,np.newaxis]
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ],3)

        with tf.variable_scope('feature_extract') as extractor:
            if reuse:
                extractor.reuse_variables()
            with tf.variable_scope('intermediate'):
                self.conv1_1 = self._conv_layer(bgr, "conv1_1")
                self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
                self.pool1 = self._max_pool(self.conv1_2, 'pool1')

                self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
                self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
                self.pool2 = self._max_pool(self.conv2_2, 'pool2')

                self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
                self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
                self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
                self.pool3 = self._max_pool(self.conv3_3, 'pool3')

                self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
                self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
                self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
                self.pool4 = self._max_pool(self.conv4_3, 'pool4')

                self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
                self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
                self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
                self.pool5 = self._max_pool(self.conv5_3, 'pool5')

            flat_pool5 = tf.contrib.layers.flatten(self.pool5)
            self.fc6 = tf.layers.dense(flat_pool5, units=4096, activation=tf.nn.relu)
            if training:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5)

            self.fc7 = tf.layers.dense(self.fc6, units=4096, activation=tf.nn.relu)
            if training:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5)

            return self.fc7
    def classify(self, feature, labels, classes):
        with tf.variable_scope('classify'):
            # the two outputs, logits is for training, probs for testing
            self.logits = tf.layers.dense(feature, units=classes, activation=None)
            self.probs = tf.nn.softmax(self.logits, name="probs")
            self.loss_pred = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits))

        return self.loss_pred, self.probs


class Model():
    def __init__(self, training):
        self.train = training
        self.vgg = Vgg16(training=self.train,vgg16_npy_path='vgg/model_file/vgg16.npy')

    def build(self, dataset_iterator, classes=2):
        (batch_X, batch_Y) = dataset_iterator.get_next()
        feat = self.vgg.feature_extract(batch_X, training=self.train)
        loss_pred, probs = self.vgg.classify(feat, batch_Y, classes=classes)

        return loss_pred, probs