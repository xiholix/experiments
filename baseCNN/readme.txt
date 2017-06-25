operatedVec/tfrecord保存一些tf格式的输入输出文件,每个example含有一个x和一个标签y
operatedVec/vocab/.*_voc保存了对应的情感词汇表
operatedVec/wordToIndex/indexVecMatrix行号代表对应的词汇索引,每行是该索引对应的词向量
operatedVec/wordToIndex/word2Index由词作为key,对应的词汇索引为value的map

2017/6/19
由于换了固态重装系统，所以该项目的数据需重新生成
1. 将所有的单标签数据放入data/rawData/binaryData文件夹下
2. 运行prepareInput.py文件下的get_vocabulary()函数得到一系列保存在data/operatedVec/vocab下的词汇表
3. 调用get_vocabulary_vec()函数得到词汇对应的词向量，结果保存在data/operatedVec/vocab_vec文件
4. 然后调用build_word_index_map()函数，得到词汇到index的map保存在data/operatedVec/wordToIndex/word2Index以及以index为行号，
   词向量为行的内容的embedding矩阵，保存在data/operatedVec/wordToIndex/indexVecMatrix
5. 调用generate_tfrecord()函数产生 data/operatedVec/tfrecord/TrainAnger.tf这个训练数据以及
   data/operatedVec/tfrecord/TestAnger.tf这个测试数据

2016/6/21
今天添加了run_epoch用来运行一个epoch，以及对于test时，指定函数的string_input_producer的num_epochs为1,用于保证结果test只运行一次。
