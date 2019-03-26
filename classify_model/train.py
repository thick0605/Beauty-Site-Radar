
# coding: utf-8
# import library
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2
from model import *
from config import *
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.learn import ModeKeys
import scipy.misc
train_graph = tf.Graph()
test_graph = tf.Graph()


def _parse_function(file, label):

    image_string = tf.read_file(file)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded, size=[256,256])
    image_resized = tf.cast(image_resized, tf.float32)
    return image_resized, label

def prepare_nn_data(mode=ModeKeys.TRAIN, num=None):
    X_filename = []
    Y_labels = []
    if mode == 'predict':
        for i in range(num):
            X_filename.append('./tmp_imgs/%02d.jpg'%(i))
            Y_labels.append(0)
    else:
      if mode == ModeKeys.TRAIN:
          img_list = open(config.train.img_list,'r')
      elif mode == ModeKeys.INFER:
          img_list = open(config.test.img_list,'r')

      for filename in img_list.readlines():
          X_filename.append(filename.split(' ')[0])
          Y_labels.append(int(filename.split(' ')[1].strip()))
  
    X_filename = tf.constant(X_filename)
    Y_labels = tf.constant(np.asarray(Y_labels))
  
    return X_filename, Y_labels

def train(data_X,data_Y):
    global train_graph

    # build model structure for training
    with train_graph.as_default():
        dataset = Dataset.from_tensor_slices((data_X,data_Y))
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=1100)
        dataset = dataset.repeat(config.train.n_epoch)
        dataset = dataset.batch(config.train.batch_size)
        iterator = dataset.make_one_shot_iterator()

        model = Model(training=True)
        loss, probs = model.build(iterator)

        var_list1 = [v for v in tf.global_variables() if 'intermediate' in v.name]
        var_list2 = [v for v in tf.global_variables() if v not in var_list1]
        
        opt1 = tf.train.AdamOptimizer(0.00001)  
        opt2 = tf.train.AdamOptimizer(0.0001)  
        grads = tf.gradients(loss, var_list1 + var_list2)  
        grads1 = grads[:len(var_list1)]  
        grads2 = grads[len(var_list1):]  
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))  
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))  
        train_step = tf.group(train_op1, train_op2)
        
        saver = tf.train.Saver(tf.global_variables())

    # start a session to train
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        while True:
            iteration += 1
            run_time = time.time()
            try:
                _, loss_value = sess.run([train_step,loss])
                if iteration % 5 == 0:
                    print('Iteration: %d Loss1: %.8f Time: %4.4f'%(iteration,loss_value,time.time()-run_time))
                if iteration % 60 == 0:
                    saver.save(sess, 'model_file/model')
            except tf.errors.OutOfRangeError:
                break

def test(data_X, data_Y):
    global test_graph
    pred_file = open('result/prediction.txt','w')
    # build model structure for training
    with test_graph.as_default():

        dataset = Dataset.from_tensor_slices((data_X, data_Y))
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(config.test.batch_size)
        iterator = dataset.make_one_shot_iterator()

        model = Model(training=False)
        loss, probs = model.build(iterator)

        # define saver
        saver = tf.train.Saver(tf.global_variables())

    # start a session to train
    with tf.Session(graph=test_graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,'model_file/model')
        iteration = 0
        hr_img_list = open(config.test.img_list,'r')
        filename_list = []
        for name in hr_img_list.readlines():
            filename_list.append(name.split(' ')[0])

        while True:
            iteration += 1
            run_time = time.time()
            try:
                pred_label = sess.run(probs)
                for i in range(len(pred_label)):
                    print(type(pred_label[i]))
                    pred_class = np.argmax(pred_label[i])
                    print(pred_class)
                    pred_file.write('%d\n'%pred_class)
                print('Iteration: %d Time: %4.4fs'%(iteration,time.time()-run_time))
            except tf.errors.OutOfRangeError:
                break
def predict(num):
    data_X, data_Y = prepare_nn_data('predict',num)
    predict_graph = tf.Graph()
    # build model structure for training
    with predict_graph.as_default():

        dataset = Dataset.from_tensor_slices((data_X, data_Y))
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(10)
        iterator = dataset.make_one_shot_iterator()

        model = Model(training=False)
        loss, probs = model.build(iterator)

        # define saver
        saver = tf.train.Saver(tf.global_variables())

    # start a session to train
    with tf.Session(graph=predict_graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,'vgg/model_file/model')
        avg_score = 0.0
        pred_label = sess.run(probs)
        for i in range(len(pred_label)):
            avg_score += pred_label[i][0]
        
        return avg_score/num
def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if sys.argv[1] == 'train':
        data_X, data_Y = prepare_nn_data()
        train(data_X,data_Y)
    elif sys.argv[1] == 'test':
        data_X, data_Y = prepare_nn_data(ModeKeys.INFER)
        test(data_X,data_Y)
    else:
        return predict(sys.argv[2])


if __name__ == '__main__':
    main()

