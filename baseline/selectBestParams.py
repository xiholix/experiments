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
@file: selectBestParams.py
@time: 17-6-14 下午10:44
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
def select_param(_trainPath, _testPath, _resultPath):
    trainY, trainX = pickle.load(open(_trainPath))
    testY, testX = pickle.load(open(_testPath))
    testY = np.array(testY)
    print(trainX.shape)
    params = {'C':[0.01, 0.05, 0.075, 0.10, 0.125, 0.15], 'class_weight':[{1:4}, {1:8}, {1:12}]}
    linearSVC = LinearSVC()
    gv = GridSearchCV(linearSVC, param_grid=params, scoring="f1_macro")
    gv.fit(trainX, trainY)
    f = open(_resultPath, "w")
    print(f)
    pickle.dump(gv, f)


def test_model(_trainPath, _testPath):
    trainY, trainX = pickle.load(open(_trainPath))
    testY, testX = pickle.load(open(_testPath))
    testY = np.array(testY)
    linearSVC = LinearSVC(C=0.075, class_weight={1:4})
    linearSVC.fit(trainX, trainY)
    predict = linearSVC.predict(testX)
    print(np.sum(predict))
    print(predict.shape)
    print(np.sum(testY))
    f1Score = f1_score(testY, predict)
    pScore = precision_score(testY, predict)
    rScore = recall_score(testY, predict)
    print(f1Score)
    print(pScore)
    print(rScore)


def load_best_model(_path):
    gv = pickle.load(open(_path))
    print('hell')


if __name__ == "__main__":
    trainPath = 'binaryVecData/Ren_CECpsTrainAnger.txt'
    testPath = 'binaryVecData/Ren_CECpsTestAnger.txt'
    # select_param(trainPath, testPath, "result/result.d2")
    # load_best_model("result/result.d2")
    test_model(trainPath, testPath)