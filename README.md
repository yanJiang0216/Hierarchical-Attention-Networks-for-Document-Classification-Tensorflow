# Hierarchical-Attention-Networks-for-Document-Classification-Tensorflow
分层注意力进行文本分类。论文地址：https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf。代码参考了https://github.com/triplemeng/hierarchical-attention-model。

语句中每个字对于此句话属于哪个类型的贡献是不一样的，比如：“今天有暴雨”，很明显，暴雨这个词决定了这句话可能是个天气预报，它的分类权重应该大，而“今天”
这个词的分类权重应该小。同样，文章中的每句话，对于文章属于哪个类型的贡献也是不一样的。所以，这篇论文的思想就是，设计字注意力以及句子注意力，为每个字以及每句话
配置不同的权重，来对文章进行分类。
此模型在使用IMDB数据时，比之前的模型提高了3%+。(详见论文)

论文使用了双向GRU，我试验了双向LSTM，以及此论文 https://arxiv.org/pdf/1810.09536.pdf 介绍的排序神经元的LSTM，未发现有明显的改进。
tensorflow实现的排序神经元的LSTM在这里：https://github.com/TieDanCuihua/ORDERED-NEURONS-INTEGRATING-TREE-STRUCTURES-INTO-RECURRENT-NEURAL-NETWORKS--tensorflow。

当分类种类较多时，可以尝试Hierarchica_softmax，以减少计算量。https://github.com/TieDanCuihua/hierarchical_softmax_tensorflow
