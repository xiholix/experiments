import numpy as np 
import datetime
from sklearn.metrics import f1_score, recall_score, precision_score
from prepareInput import cecEmotionList
from config import FLAGS, filterWindow
def get_true_label(path):
    with open(path) as f:
        labels = []
        line = f.readline()
        while line:
            label = int(line.split()[0])
            labels.append(label)
            line = f.readline()
    
    # print labels
    print len(labels)
    return labels


def get_true_label_train(path):
    with open(path) as f:
        labels = []
        line = f.readline()
        while line:
            label = int(line.strip().split()[-1])
            labels.append(label)
            line = f.readline()
    
    # print labels
    print len(labels)
    return labels

def get_predict_label(path):
    with open(path) as f:
        labels = []
        line = f.readline()
        while line:
            label = int(line.strip().split()[0])
            labels.append(label)
            line = f.readline()
        print len(labels)
    return labels


def evaluate(emotion, f):
    true_label = np.array(get_true_label("data/rawData/binaryData/Ren_CECpsTest"+emotion+".txt"))
    predict_labels = np.array(get_predict_label("data/result/"+emotion+".pre"))[:6153]
   
    print "********************************************************"
    print emotion
    f.write(emotion+"\n")
    f1 = f1_score(true_label, predict_labels)
    rs = recall_score(true_label, predict_labels)
    pr =  precision_score(true_label, predict_labels)
    f.write(str(f1)+"\n"+str(rs)+"\n"+str(pr)+"\n\n\n")
    
    # print np.sum(true_label==predict_labels[:6153])
    # print np.sum(true_label)
    # print np.sum(predict_labels[:6153])


def evaluate_train(emotion, f):
    true_label = np.array(get_true_label_train("data/result/"+emotion+".pre"))[:6153]
    predict_labels = np.array(get_predict_label("data/result/"+emotion+".pre"))[:6153]
   
    print "********************************************************"
    print emotion
    f.write(emotion+"\n")
    f1 = f1_score(true_label, predict_labels)
    rs = recall_score(true_label, predict_labels)
    pr =  precision_score(true_label, predict_labels)
    f.write(str(f1)+"\n"+str(rs)+"\n"+str(pr)+"\n\n\n")
    
    # print np.sum(true_label==predict_labels[:6153])
    # print np.sum(true_label)
    # print np.sum(predict_labels[:6153])

def record_params(f):
    f.write("*****************************************************************\n")
    f.write(str(datetime.datetime.now())+"\n")
    FLAGS.filterNums
    for key, value in FLAGS.__flags.items():
        oneLine = str(key)+"\t"+str(value)+"\n"
        f.write(oneLine)
    f.write("filterWindow\t"+str(filterWindow))
    f.write("\n\n")

def fscore(true_labels, predict_labels):
    a = true_labels.reshape((-1))
    b = predict_labels.reshape((-1))
    print 'f1 score %f'%(f1_score(a, b))

if __name__ == "__main__":
    f = open(FLAGS.evaluateResultPath, "a")
    record_params(f)
    for emotion in cecEmotionList:
        evaluate_train(emotion, f)