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
@file: config.py
@time: 17-6-17 上午8:10
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
filterWindow = [2, 3, 4]
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("filterNums", 100, "the filter num")
flags.DEFINE_integer("vecDim", 300, "the dimension of wordvec")
flags.DEFINE_integer("batchSize", 32, "the size of one batch")
flags.DEFINE_integer("sentenceLength", 90,"the length of one sentence")
flags.DEFINE_integer("lr", 1e-4, "the learning rate")
flags.DEFINE_string("savedir", "data/withDropout/", "the path of event file")
flags.DEFINE_float("keepProb", 0.6, "keepProb")
flags.DEFINE_string("evaluateResultPath", "data/evaluateResult.txt", "the evaluated result path")
flags.DEFINE_string("resultDir", "data/result/", "the result dir")
if __name__ == "__main__":
   print(FLAGS.filterNums)
#    print ('\n'.join(['%s:%s' % item for item in FLAGS.__dict__.items()]) )
   print(FLAGS.__flags)