# vehicle-signal-lights-states-classification
TUD master thesis project, vehicle signal lights states classification based on image, pytorch.

### reference:

26.06
测试了一下不用pre-trained model，照样overfitting，freezed weight也无用，而且好像不是因为模型太大而是因为模型太小而没有学到位？

今天测试下普通cnn的效果，看看什么时候变成 underfitting

25.06
训练的valid loss比不训练还大，还一直上升，我感觉是网络结构出了问题，但是training loss一直在降，很奇怪，好在找到了一个类似的repo

tips: toTensor() 会把值归一化到 [0,1]，之后的normalize是会把图像归一化到更合适的位置

24.06
尝试解决overfitting的问题

一个小坑：dropout 的值表示被舍弃的比例，默认不舍弃所以为 0.0

尝试了：
减少LSTM的层的dim，没啥用，收敛慢下来了但是还是过拟合
试了下正则化，导致模型不收敛了（weight decay == 10/0.1）
加大了数据量，好像也没啥用 ⇒ 也许可以检查下怎么取数据的
resnet freeze:  全部冻结好像会导致不拟合，原因不明，目前保留了最后一层，勉强能收敛，而且之前以为模型简单所以要设置一个更小的 lr，结果其实用和不freeze时一样的就可以了

batch size 越小越好：使得 generalization 更好
https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network 

speed up the dataloader
https://tianws.github.io/skill/2019/08/27/gpu-volatile/ 
23.06
ok终于能收敛了，现在train loss可以一直减下去，最低到0.02，但是valid loss一直减不下去，过拟合了

TODO：
加大数据量：数据不平衡咋办？：https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/5
修改sampler

https://discuss.pytorch.org/t/how-to-avoid-overfitting-in-pytorch/17581
dropout
把resnet freeze

或许再加个正则化项？
https://medium.com/@diazagasatya/will-dropout-regularization-prevents-your-model-to-overfit-11afa10cd4e0

这里说 batch_normalization 可以防过拟合：
https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
要不打印一下最后resnet的pool输出？
然后加一个batch norm层

data augmentation（尤其是对于少量数据的部分），但是数据都用不了全部，等可以用taurus再试试吧

22.06
训练了一晚上，好像training loss减不下去，但是应该不是已经拟合好了，因为还持续比较高的位置，应该是underfitting了
TODO: 
先看看数据有没有错误：做个可视化，尤其是在输入到lstm之前
果然是这里出了问题，batch_size*len 直接 reshape成 len*batch_size了，搞得每个序列里的图片其实来自不同batch.

再看看能不能在极小数据集上收敛，好想法，主要看看这个模型有没有错误
试了，确实有用，在没debug前连每个类别3个视频都收敛不了，修改之后一个epoch就好很多

听说batch_size设得太小也有问题 ⇒ 这个得在其他机器上测试了
试下 taurus？好像更方便



TODO: 设置一下dataloader 的num-worker 和 pin-mem
加一个lr_scheduler

GPU 优化：prefechter:     https://zhuanlan.zhihu.com/p/66145913
gpu利用率上下浮动：    https://blog.csdn.net/Strive_For_Future/article/details/98872216

lr_scheduler:
http://www.spytensor.com/index.php/archives/32/

batch norm 来加速训练？ 
对lstm: https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839
https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py <== 每个lstm layer之间的bn，很麻烦啊。。。

通常来说 batch norm用在提取特征的卷积层：
https://discuss.pytorch.org/t/batch-normalization-of-linear-layers/20989/9

设置的batch size太小有可能降低性能
https://stackoverflow.com/questions/57457817/adding-batch-normalization-decreases-the-performance

解释： https://www.youtube.com/watch?v=nUUqwaxLnWs
https://www.youtube.com/watch?v=-5hESl-Lj-4

adam等优化器的解释，并不是调整lr
https://zhuanlan.zhihu.com/p/32626442

