from prepareInput import reader_tfrecoder
import numpy as np
import tensorflow as tf
import pickle
path = "data/operatedVec/tfrecord/TestAnger.tf"

def compare():
    testx1, testy1 = reader_tfrecoder(path)
    # testx2, testy2 = reader_tfrecoder(path)
    
    sess = tf.Session()

    init = tf.global_variables_initializer()
    l = tf.local_variables_initializer()
    sess.run(init)
    sess.run(l)
   
    tf.train.start_queue_runners(sess=sess)
    x1 = sess.run(testx1)
    # y1 = sess.run(testy1)
    # x2 = sess.run(testx2)
    print(x1.shape)
    # pickle.dump(x1, open("data/testCompare/x1.d", "wb"))
    x2 = pickle.load( open("data/testCompare/x1.d"))
    print(x2[0, :10])
    print(x1[0, :10])
    r = np.sum(x1-x2)
    print("difference")
    print(r)
    # print(x2)
    # print(y1)
    # y1 = sess.run(testy1)
    # y1 = sess.run(testy1)
    # print(y1)

if __name__ == "__main__":
    compare()