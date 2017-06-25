import numpy as np 

from prepareInput import cecEmotionList
from config import FLAGS

def generateResult():
    resultPath = "data/generatedResult.d"
    f = open(resultPath, "wb")
    nameToFile = []
    for emotion in cecEmotionList:
        path = FLAGS.resultDir+emotion+".pre"
        nameToFile.append(open(path))
    results = []
    sign = False
    while True:
        oneProb = []
        for i in xrange(8):
            line = nameToFile[i].readline()
            if not line:
                sign = True
                break
            prob = float(line.strip().split()[1])
            oneProb.append(prob)

        if sign:
            break
        results.append(oneProb)
    
    for i in xrange(len(results)):
        result = np.array(results[i])
        index = np.argsort(-result)
        oneLine = ""
        for i in xrange(4):
            # if i==0:
            #     oneLine += cecEmotionList[index[i]]+' '
            #     continue
            if result[index[i]] >= 0.5:
                oneLine += cecEmotionList[index[i]]
            else:
                oneLine += "none"
            if i < 3:
                oneLine += ' '
            else:
                oneLine += '\n'
        f.write(oneLine)


if __name__ == "__main__":
    generateResult()


