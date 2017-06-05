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
@file: oprData.py
@time: 17-6-2 下午9:01
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import codecs
import numpy as np
import pickle

cecEmotion = {'Anxiety':0, 'Love':1, 'Joy':2, 'Sorrow':3, 'Expect':4, 'Anger':5, 'Surprise':6, 'Hate':7}
cecEmotionList = ['Anxiety', 'Love', 'Joy', 'Sorrow', 'Expect', 'Anger', 'Surprise', 'Hate']
vecSize = 300
def get_binary_algorithm_data():
    '''
    根据filePaths中的文件名,将多标签数据转换成相应的多个单标签数据文件.
    现在的实现方式中只处理Ren_CECps数据集,所以Emotion是cecEmotionList中的情感
    :return:
    '''
    path = 'data'
    outPath = 'binaryData'
    # filePaths = os.listdir(path)
    filePaths = ['Ren_CECpsTest.txt', 'Ren_CECpsTrain.txt']
    for f in filePaths:
        fPath = path + os.sep + f
        outFile = []
        for e in cecEmotionList:
            tPath = outPath + os.sep + f.split('.')[0] + e + '.txt'
            outFile.append( codecs.open(tPath, 'w') )   #加上utf-8会报错,不知道为什么

        with open(fPath) as f:
            line = f.readline()
            while line:
                words = line.split()
                emotions = words[:4]
                contents = ' '.join(words[4:])


                sign = ['0']*8
                for e in emotions:
                    if cecEmotion.has_key(e):
                        sign[cecEmotion[e]] = '1'

                for s, writeFile in zip(sign, outFile):
                    writeFile.write(s+' '+contents+os.linesep)

                line = f.readline()
        for o in outFile:
            o.close()


def get_emotion_set():
    '''
    根据path文件中的数据集得到情感标签的种类
    :return:
    '''
    path = 'data/Ren_CECpsTrain.txt'
    f = open(path)
    emotion = set()
    line = f.readline()
    while line:
        emotions = line.split()[:4]
        for e in emotions:
            emotion.add(e)
        line = f.readline()
    print(emotion)


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



def get_binary_vec_data():
    '''
    将单标签数据文件中数据用对于的词向量表示,句子的词向量由句子中在词汇表中的词汇的向量的加和
    对应的处理后的文件可以用path = 'binaryVecData/Ren_CECpsTrainHate.txt'
    labels, datas = pickle.load(open(path))得到对应的labels和datas数据
    :return:
    '''
    path = 'binaryData'
    resultPath = 'binaryVecData'
    filePaths = os.listdir(path)
    word_vec = pickle.load(open('data/word_vec_cec.d'))
    for f in filePaths:
        fPath = path + os.sep + f
        result = resultPath + os.sep + f
        print(result)
        with open(fPath) as reader:
            line = reader.readline()
            labels = []
            datas = []
            while line:
                words = line.split()
                label = words[0]
                labels.append(int(label))
                words = words[1:]

                oneData = np.array([0.0]*vecSize)
                for w in words:
                    if word_vec.has_key(w):
                        oneData += word_vec[w]
                # print(oneData)
                datas.append(oneData)
                line = reader.readline()
        datas = np.stack(datas)
        pickle.dump([labels, datas], open(result, 'wb'))


def get_vocabulary():
    '''
    根据path得到数据的词汇表
    :return:
    '''
    path = 'data/Ren_CECpsTrain.txt'
    vocab = set()

    with open(path) as f:
        line = f.readline()
        while line:
            words = line.split()[4:]
            for w in words:
                vocab.add(w)
            line = f.readline()
    vocab = list(vocab)
    print(len(vocab))
    return vocab


def get_vocabulary_vec():
    '''
    根据得到的词汇表将该词汇表中的词对应的词向量保存到文件中,避免每次读取那么大的词向量
    :return:
    '''
    vocab = get_vocabulary()
    word_vec = load_bin_vec('data/allTextOne20151120SkipCount5Size300.bin', vocab)
    pickle.dump(word_vec, open('data/word_vec_cec.d', 'wb'))



def test_binary_vec_data():
    path = 'binaryVecData/Ren_CECpsTrainHate.txt'
    labels, datas = pickle.load(open(path))
    print(len(labels))
    print(len(datas))
    print(type(datas))
    # datas = np.stack(datas)
    print(datas.shape)
    print('the ratio between label 1 and 0 {0}'.format(np.sum(labels)/len(labels)))
    print('the num of label 1 is: %d'%(np.sum(labels)))


def overlook_one_dataset(_path):
    labels, datas = pickle.load(open(_path))
    print('the ratio between label 1 and 0 {0}'.format(np.sum(labels) / len(labels)))
    print('the num of label 1 is: %d' % (np.sum(labels)))


def overlook_the_dataset():
    '''
    use to view the imbalance of these emotion labels
    :return:
    '''
    trainOrTest = 'Train'   #element of ('Train', 'Test'), be careful the case
    dir = 'binaryVecData'
    filePaths = os.listdir(dir)

    for f in filePaths:
        filePath = dir + os.sep + f
        # print(filePath)
        if trainOrTest in f:
            print('the result of %s'%filePath)
            overlook_one_dataset(filePath)


def discard_mostclass_data():
    '''
    将binaryVecData文件夹下的训练数据的多数类的数据丢弃一些,形成与少数类一样多的平衡的训练数据用于训练
    :return:
    '''
    baseDir = 'binaryVecData'
    outputDir = 'data/discardMostClassData'
    filePaths = os.listdir(baseDir)

    for f in filePaths:
        if 'Train' not in f:
            continue
        filePath = baseDir + os.sep + f
        outputPath = outputDir + os.sep + f
        labels, datas = pickle.load(open(filePath))
        numTheLeastLabels = np.sum(labels)
        indiceSort = np.argsort(labels)
        indices = indiceSort[:numTheLeastLabels]
        mostLabelsIndices = indiceSort[numTheLeastLabels:]
        np.random.shuffle(mostLabelsIndices)   #np.random.shuffle直接对参数操作,没有返回值
        indices = np.concatenate((indices, mostLabelsIndices[:numTheLeastLabels]))
        np.random.shuffle(indices)
        # print(type(labels))  #labels的类型是list,在本函数处理后将会转换为array
        # print(type(datas))
        labels = np.array(labels)
        labels = labels[indices]
        datas = datas[indices]
        print(datas.shape)
        pickle.dump((labels, datas), open(outputPath, 'w'))


if __name__ == "__main__":
    # get_binary_algorithm_data()
    # get_emotion_set()
    # get_binary_vec_data()
    # get_vocabulary()
    # get_vocabulary_vec()
    # test_binary_vec_data()
    # overlook_the_dataset()
    discard_mostclass_data()