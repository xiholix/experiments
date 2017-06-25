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
@file: baselineSVM.py
@time: 17-6-2 下午10:31
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

from oprData import *


def base_svm(_index):
    labels, datas = get_data_by_emotion(cecEmotionList[_index], True)
    svc = SVC()
    print('begin train')
    startTime = time.time()
    svc.fit(datas, labels)
    testLabels, testDatas = get_data_by_emotion(cecEmotionList[0], False)
    endTime = time.time()
    print('elapse time is :')
    print(endTime-startTime)

    print('begin test')
    predict = svc.predict(testDatas)
    score = accuracy_score(testLabels, predict)
    print(score)
    print('predict 1 ratio')
    print( np.sum(predict)/len(predict) )
    resultPath = 'data/result/'+cecEmotionList[_index]
    pickle.dump((predict, svc), open(resultPath, 'w'))


def use_svm():
    for i in xrange(0 , len(cecEmotionList)):
        base_svm(i)

if __name__ == "__main__":
    use_svm()