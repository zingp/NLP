# N元模型示例程序

提示：得到源码后，先执行跟目录下的 admin/run.sh 进入docker容器服务。然后在容器内进行下述步骤。

该演示项目基于开源项目[kenlm](https://github.com/kpu/kenlm)。

## 生成模型文件
```
./gen_model.sh
```

数据文件 data/ngrams.train.tokens 是训练n-grams模型的输入数据。程序执行结束后，将得到 ngrams.arpa.gz文件。

## 计算一个句子的出现概率
```
python lm.py Test.test_prob
```

## 计算一个句子的困惑度
```
python lm.py Test.test_perplexity
```

![](/assets/images/ngrams-lm.png)
