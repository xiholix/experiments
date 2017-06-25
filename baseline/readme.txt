2017/6/5:
新建了这个readme文件,以后把每次编写的代码总结一下放到这个文件中,以及待解决的问题,或者是比较好的思路.方便自己以后使用这些代码

在用codecs写入本数据集是会因为open时候加了utf8参数报错,取掉该编码就正确 ???待解决
今天采用丢弃一部分大部分类的数据,使用svm来训练,最后来预测.

***binaryVecData***中的labels是list类型,但是datas是array类型
使用svm默认参数在丢弃大部分类的数据集上训练的结果并不好.几乎将大部分结果分为0


2017/6/19:
今天将电脑装了ssd，并且重新安装了操作系统，导致数据丢失了，所以现在重新对项目构建数据。
1. 使用oprData文件中对get_binary_algorithm_data()函数将文件夹下的多标签数据转换成单标签数据，保存到binaryData目录下
2. 然后使用get_vocabulary_vec()函数的到data文件夹下的word_vec_cec.d这个词语到词向量的转换
3. 接着使用get_binary_vec_data()函数来得到binaryVecData文件夹下各个文件内容用词向量转换的结果