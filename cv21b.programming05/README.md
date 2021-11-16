# README.md

## 实验目的

进行手写签名识别

## 使用网络

带有注意力机制的seq2seq网络

## 使用方法

首先，可以运行convert.py将train文件夹分成训练集与测试集以及生成所有识别字符的列表，保存在train.txt与test.txt以及char...txt中，格式可以自己修改成对应的格式，然后运行命令python train.py --train_list train.txt --eval_list test.txt --model ./model/crnn/ 进行训练，然后修改路径运行interference.py进行识别。

## Reference

* [caffe_ocr](https://github.com/senlinuc/caffe_ocr)

* [PyTorch Tutorials >  Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

