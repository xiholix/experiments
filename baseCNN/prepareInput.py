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
@file: prepareInput.py
@time: 17-6-17 上午8:34
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import pickle
import numpy as np
import re
from config import FLAGS

cecEmotionList = ['Anxiety', 'Love', 'Joy', 'Sorrow', 'Expect', 'Anger', 'Surprise', 'Hate']


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        count = 0
        for line in xrange(vocab_size):
            if count%10000==0:
                print (count)
            count += 1
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_vocabulary_from_file(path):
    d = set()
    with open(path) as f:
        line = f.readline()
        while line:
            words = line.split()[1:]
            for w in words:
                d.add(w)
            line = f.readline()
    voc = list(d)
    return voc


def get_vocabulary():
    '''
    根据不同的文件得到不同的文件内容的词汇表
    词汇的内容放在data/operatedVec/vocab/下的对应的_voc文件
    :return:
    '''
    trainPath = "data/rawData/binaryData/Ren_CECpsTrainAnger.txt"
    resultDir = "data/operatedVec/vocab"
    for w in cecEmotionList:               #好像所有的训练集的词汇表应该都是一样的
        newTrainPath = re.sub('Anger', w, trainPath)
        resultFileName = w+"_voc"
        newResultPath = resultDir+"/"+resultFileName
        voc = get_vocabulary_from_file(newTrainPath)
        pickle.dump(voc, open(newResultPath, "wb"))


def get_vocabulary_vec():
    '''
    根据词汇表得到对应的词汇表的词向量,结果保存在dat/operatedVec/vocab_vec文件
    :return:
    '''
    resultPath = "data/operatedVec/vocab_vec"
    vocabularyPath = "data/operatedVec/vocab/Anger_voc"
    vecPath = "data/rawData/allTextOne20151120SkipCount5Size300.bin"
    vocab = pickle.load(open(vocabularyPath))
    word2Vec = load_bin_vec(vecPath, vocab)
    pickle.dump(word2Vec, open(resultPath, "wb"))
    # print(word2Vec)


def build_word_index_map():
    '''
    词汇到index的map保存在data/operatedVec/wordToIndex/word2Index
    对应的index作索引代表的词向量作为一行的词向量矩阵保存在data/operatedVec/wordToIndex/indexVecMatrix
    :return:
    '''
    vocabularyPath = "data/operatedVec/vocab/Anger_voc"
    vocab = pickle.load(open(vocabularyPath))
    wordMapPath = "data/operatedVec/vocab_vec"
    wordMap = pickle.load(open(wordMapPath))
    path = "data/operatedVec/wordToIndex/word2Index"
    matrixPath = "data/operatedVec/wordToIndex/indexVecMatrix"
    matrix = np.zeros(shape=(len(vocab)+1, FLAGS.vecDim))   #0号索引是留给没在词汇表出现的词语
    word2Index = {}
    i = 1
    for v in vocab:
        word2Index[v] = i
        if wordMap.has_key(v):
            matrix[i] = wordMap[v]
        i += 1
    pickle.dump(word2Index, open(path, "wb"))
    pickle.dump(matrix, open(matrixPath, "wb"))


def opr_one_line(line, wordToIndex, length):
    '''
    把一行按照wordToIndex转换成index表示
    :param line:
    :param wordToIndex:
    :param length:
    :return:
    '''
    words = line.split()
    label = int(words[0])
    oneLine = [0]*length
    words = words[1:]
    i = 0
    for w in words:
        if wordToIndex.has_key(w):
            oneLine[i] = wordToIndex[w]
        else:
            oneLine[i] = 0
        i += 1
    return label, oneLine


def generate_tfrecord(emotion):
    index = get_balance_data_index(emotion)
    trainPath = "data/rawData/binaryData/Ren_CECpsTrain"+emotion+".txt"
    testPath = "data/rawData/binaryData/Ren_CECpsTest"+emotion+".txt"
    path = "data/operatedVec/wordToIndex/word2Index"
    resultPath = "data/operatedVec/tfrecord/Train"+emotion+".tf"
    testResultPath = "data/operatedVec/tfrecord/Test"+emotion+".tf"
    writer = tf.python_io.TFRecordWriter(resultPath)

    wordToIndex = pickle.load(open(path))
    with open(trainPath) as f:
        line = f.readline()
        i = 0
        while line:
            if i in index:
               label, data = opr_one_line(line, wordToIndex, FLAGS.sentenceLength)
               xFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=data))
               yFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
               features = tf.train.Features(feature={
                   'x':xFeature,
                   'y':yFeature
               })
               example = tf.train.Example(features=features)
               writer.write(example.SerializeToString())
            # print(data)
            i += 1
            line = f.readline()

    writer = tf.python_io.TFRecordWriter(testResultPath)

    wordToIndex = pickle.load(open(path))
    with open(testPath) as f:
        line = f.readline()
        i = 0
        while line:
               label, data = opr_one_line(line, wordToIndex, FLAGS.sentenceLength)
               xFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=data))
               yFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
               features = tf.train.Features(feature={
                   'x':xFeature,
                   'y':yFeature
               })
               example = tf.train.Example(features=features)
               writer.write(example.SerializeToString())
            # print(data)
               i += 1
               line = f.readline()


def reader_tfrecoder(path, isTrain=False):
    pathList = [path]
    if isTrain:
        queue = tf.train.string_input_producer(pathList)
    else:
        queue = tf.train.string_input_producer(pathList, 2)
    reader = tf.TFRecordReader()
    reader.reset()
    _, example = reader.read(queue)
    example = tf.parse_single_example(example, features={
        'x':tf.FixedLenFeature([FLAGS.sentenceLength], tf.int64),
        'y':tf.FixedLenFeature([1], tf.int64)
    })
    if isTrain:
        batchData = tf.train.shuffle_batch([example['x'], example['y']],
                                           FLAGS.batchSize,
                                           FLAGS.batchSize*5,
                                           FLAGS.batchSize)
    else:
         batchData = tf.train.batch([example['x'], example['y']],
                                           FLAGS.batchSize,
                                           1,
                                           FLAGS.batchSize*5,
                                           )
    return batchData


def get_balance_data_index(emotion):
    path = "data/rawData/binaryData/Ren_CECpsTrain"+emotion+".txt"
    truely_index = []
    false_index = []
    i = 0
    with open(path) as f:
        line = f.readline()
        while line:
            words = line.split()
            if words[0]=="1":
                truely_index.append(i)
            else:
                false_index.append(i)
            i += 1
            line = f.readline()
    print (len(truely_index))
    print (len(false_index))
    lengths = len(truely_index)
    false_index = np.array(false_index)
    np.random.shuffle(false_index)
    false_index = false_index[:lengths]
    truely_index = np.array(truely_index)
    index = np.concatenate((truely_index, false_index))
    np.random.shuffle(index)
    return index


if __name__ == "__main__":
    # get_vocabulary_vec()
    # build_word_index_map()
    # generate_tfrecord()
    # path = "data/operatedVec/tfrecord/TrainAnger.tf"
    # batchData = reader_tfrecoder(path)
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # tf.train.start_queue_runners(sess)
    # print("start")
    # b = sess.run(batchData)
    # print(len(b[0]))
    # print(b[0][3])
    # get_vocabulary()
    for emotion in cecEmotionList:
        print(emotion)
        generate_tfrecord(emotion)
    
  