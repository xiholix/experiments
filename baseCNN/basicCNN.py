# -*-coding:utf8-*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
'''
@version: ??
@author: xiholix
@contact: x123872842@163.com
@software: PyCharm
@file: basicCNN.py
@time: 17-6-17 上午8:22
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pickle
import tensorflow as tf
import numpy as np
import functools
import time
from sklearn.metrics import f1_score
from config import FLAGS, filterWindow
from prepareInput import reader_tfrecoder, cecEmotionList
from evaluated import fscore
import os





def double_wrap(function):

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args)==1 and len(kwargs)==0 and callable(args[0]):
            return function(args[0])
        else:
            def fn(decoratorFunction):
                return function(decoratorFunction, *args, **kwargs)
            return fn
    return decorator

@double_wrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_'+function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.name_scope(scope, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def get_weight(_shape, _name):
    v = tf.get_variable(_name, _shape)
    return v


def get_bias(_shape, _name):
    return tf.get_variable(_name, _shape)


class BasicCNN(object):
    def __init__(self, batchX, batchY, isTrain=False):
        self.batchX = batchX
        self.batchY = batchY
        self.isTrain = isTrain
        embedding = pickle.load(open("data/operatedVec/wordToIndex/indexVecMatrix"))
        embedding = np.array(embedding, np.float32)
        print(embedding.shape)
        self.embeddingMatrix = tf.get_variable("embedding", initializer=embedding, dtype=tf.float32)
        self.convolution
        self.fullConnect
        self.predict
        self.loss
        self.accuracy
        self.optimize

    @define_scope("embedding")
    def embedding(self):
        em = tf.nn.embedding_lookup(self.embeddingMatrix, self.batchX)
        em = tf.expand_dims(em, axis=-1)
        if self.isTrain:
            em = tf.nn.dropout(em, FLAGS.keepProb)
        else:
            print("not  is dropout********************************88")
        return em

    @define_scope("convolution")
    def convolution(self):
        stepResult = []
        for step in filterWindow:
            shape = [step, FLAGS.vecDim, 1, FLAGS.filterNums]
            w = get_weight(shape, "convw"+str(step))
            b = get_bias([FLAGS.filterNums], "convb"+str(step))
            result = tf.nn.conv2d(self.embedding, w, [1,1,1,1], padding="VALID")+b
            result = tf.nn.tanh(result)
            # shape batch*(90-step+1)*1*FLAGS.filterNums

            result = tf.nn.max_pool(result, [1, FLAGS.sentenceLength-step+1, 1, 1], [1,1,1,1], padding="VALID")
            #shape batch*1*1*FLAGS.filterNums
            result = tf.reshape(result, (FLAGS.batchSize, -1) )
            stepResult.append(result)
        
        result = tf.concat(stepResult, -1)
        #shape batch*(FLAGS.filterNums*len(filterWindow))
        return result

    @define_scope("fullConnect")
    def fullConnect(self):
        w = get_weight([FLAGS.filterNums*len(filterWindow), 1], "fullw")
        b = get_bias([1], "fullb")
        x = self.convolution
        if self.isTrain:
            x = tf.nn.dropout(x, FLAGS.keepProb)
        else:
            print("not  is dropout********************************88")
        r = tf.matmul(x, w) + b 
        return tf.nn.sigmoid(r)

    @define_scope("predict")
    def predict(self):
        fullConnect = tf.reshape(self.fullConnect, (FLAGS.batchSize,))
        result = tf.cast(tf.greater_equal(fullConnect, 0.5), dtype=tf.int32)
        return result

    @define_scope("loss")
    def loss(self):
        labels = tf.cast(self.batchY, dtype=tf.float32)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.fullConnect)
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.batchY, logits=self.fullConnect)
        loss = (tf.reduce_sum(labels*tf.log(self.fullConnect))+tf.reduce_sum(1-labels)*(tf.log(1-self.fullConnect)))/(-FLAGS.batchSize)
        # return tf.reduce_mean(loss)
        return loss
    @define_scope("optimize")
    def optimize(self):
        trainOp = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(self.loss)
        return trainOp

    @define_scope("accuracy")
    def accuracy(self):
        accuracy = tf.cast(tf.equal(tf.cast(self.batchY, dtype=tf.int32), self.predict), dtype=tf.float32)
        return tf.reduce_mean(accuracy)


def run_epoch(sess, model, epochSize, isTrain=False, outputPath=None):
    startTime = time.time()
    fetch = {
        'predict' : model.predict,
        'label' : model.batchY,
        'loss' : model.loss,
        'accuracy' : model.accuracy,
        'prob':model.fullConnect
    }
    if isTrain:
        fetch['optimizer'] = model.optimize
        print("ot***************************************")
    else:
        print("not _________________________________________---")
    print(fetch)
    f = outputPath
    if f:
        f = open(f, "a")
    for i in xrange(epochSize):
        result = sess.run(fetch)
        # fscore(result['label'], result['predict'])
        # if i%100 == 0:
        #     print (np.mean(result['loss']) )
        #     print (result['accuracy'])
            
        if f:
            for j in xrange(len(result['predict'])):
                f.write(str(result['predict'][j])+"\t"+str(result['prob'][j,0])+"\t"+str(result['label'][j, 0])+"\n")

    


def test():
    path = "data/operatedVec/tfrecord/TrainAnger.tf"
    batchX, batchY = reader_tfrecoder(path)
    cnn = BasicCNN(batchX, batchY)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    # print(sess.run(cnn.predict))
    # print(sess.run(cnn.loss))
    # print(sess.run(cnn.batchY))
    # print(sess.run(cnn.fullConnect))
    run_epoch(sess, cnn, 1000, True, "data/result/anger.predict")


def use_supervisor(isTrain=False, emotion="Anger"):
    if not os.path.exists(FLAGS.savedir+"/"+emotion):
        os.mkdir(FLAGS.savedir+"/"+emotion)
    with tf.Graph().as_default() as g:
        path = "data/operatedVec/tfrecord/Train"+emotion+".tf"
        batchX, batchY = reader_tfrecoder(path, True)
        with tf.variable_scope("model"):
            cnnModel = BasicCNN(batchX, batchY, True)
        testPath = "data/operatedVec/tfrecord/Test"+emotion+".tf"
        testX, testY = reader_tfrecoder(testPath)
        with tf.variable_scope("model", reuse=True):
            testCNN = BasicCNN(testX, testY, False)

        sv = tf.train.Supervisor(logdir=FLAGS.savedir+"/"+emotion, save_model_secs=120)
       
        with sv.managed_session() as sess:
            for i in xrange(200000):
                run_epoch(sess, cnnModel, 10, True)
                print(i)
                if i==4:
                    run_epoch(sess, testCNN, 400, False, "data/result/"+emotion+".pre")
                    break

if __name__ == "__main__":
    for emotion in cecEmotionList:
        print(emotion)
        use_supervisor(True, emotion)
