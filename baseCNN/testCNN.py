from basicCNN import *
import shutil
def use_supervisor(emotion="Anger"):
    with tf.Graph().as_default() as g:
        testPath = "data/operatedVec/tfrecord/Test"+emotion+".tf"
        testX, testY = reader_tfrecoder(testPath)
        with tf.variable_scope("model"):
            testCNN = BasicCNN(testX, testY, False)
        sv = tf.train.Supervisor(logdir=FLAGS.savedir+"/"+emotion, save_model_secs=120)      
        with sv.managed_session() as sess:
                 run_epoch(sess, testCNN, 400, False, "data/result/"+emotion+".pre")


def del_history_result():
    path = 'data/result/'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


if __name__ =="__main__":
    del_history_result()
    for emotion in cecEmotionList:
        print(emotion)
        use_supervisor(emotion)
                    