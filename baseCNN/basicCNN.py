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
import tensorflow as tf
import functools
from config import FLAGS

filterWindow = [2, 3, 4]


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


    def embedding(self):
        pass

    def convolution(self):
        pass

    def maxPool(self):
        pass

    def fullConnect(self):
        pass

    def predict(self):
        pass

    def optimize(self):
        pass