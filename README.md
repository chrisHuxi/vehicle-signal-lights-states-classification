# vehicle-signal-lights-states-classification
TUD master thesis project, vehicle signal lights states classification based on image, pytorch.

### reference:
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

