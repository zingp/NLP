# -*- encoding:utf8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import json
import pickle
import os
import zipfile
import heapq
import time
import itertools
import threading
import numpy.random as random
import sys
import scipy
import scipy.sparse as sparse
import math
from multiprocessing.pool import ThreadPool
from collections import Counter
from Queue import Queue
from numpy import float32 as REAL

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('glove')

from utils import SaveAndLoad

"""
# Introduction

A simple code version for glove

# Reference
[1] https://github.com/stanfordnlp/GloVe
[2] https://github.com/hans/glove.py/blob/master/glove.py
"""




class Glove(SaveAndLoad):
    def __init__(self, corpus=None, size=100, alpha=0.05, 
                 window=10, min_count=10, seed=1, iters=5, 
                 symmetric=True, sort_vocab=True, 
                 use_adagrad=True, merge_func='mean', tokenizer=None):
        """
        glove模型，corpus表示训练语料，输入的可以是string或句子的list，也可以是list(list())
        如`['I am happy.', 'I am unhappy.']`或`[['I', 'am', 'happy'], ['I', 'am', 'unhappy', '.']]`

        `size` 词向量维度，也就是训练的时候隐变量的维度
        `alpha` 初始学习率，学习过程中随着呈线性下降趋势
        `window` 一句话中的当前词与上下文词的最大距离，指单侧词窗（与word2vec不同，此处指左侧）
        `min_count` 用于过滤的最小词频
        `seed` 随机数种子
        `iters` 迭代次数
        `sort_vocab` 表示是否对id2word按词频从高到低排序
        `use_adagrad` 表示是否使用online adagrad更新参数，否则是sgd
        `merge_func` 表示使用什么方式将中心词权重矩阵与上下文权重矩阵合并，方法有mean、sum、concat
        `tokenizer` 表示传入一个tokenizer的function，默认按空格分词
        ``

        """
        self.size =size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.random = random.RandomState(seed)
        self.epochs = iters
        self.symmetric = symmetric
        self.use_adagrad = use_adagrad
        self.merge_func = merge_func
        self.vocab = {}
        self.id2word = {}
        self.cooccurrences = None
        self.sort_vocab = sort_vocab
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.W = None
        self.b = None
        self.grad_square_W = None
        self.grad_square_b = None
        self.W_norm = None
        self.vocab_size = 0
        if self.tokenizer is None:
            self.tokenizer = lambda sentence: sentence.strip().split()
        if self.corpus is not None:
            if len(self.corpus) < 1:
                raise ValueError('The size of the corpus must not be empty, do you mean `None`?')
            if isinstance(self.corpus[0], basestring):
                self.corpus = [self.tokenizer(_) for _ in self.corpus]
            elif isinstance(self.corpus, basestring):
                self.corpus = [self.tokenizer(self.corpus)]
            self.build_vocab(self.corpus)
            self.build_cooccurance_matrix(self.corpus)
            self.train()

    def build_vocab(self, corpus):
        """
        构建整个语料的词典
        """
        self.vocab = Counter()
        for sentence in corpus:
            self.vocab.update(sentence)
        vocab_list = self.vocab.iteritems()
        if self.sort_vocab:
            vocab_list = sorted(vocab_list, key=lambda k: -k[1])
        self.vocab = {word:(i, freq) for (i, (word, freq)) in enumerate(vocab_list)}
        self.id2word = {i:(word, freq) for (i, (word, freq)) in enumerate(vocab_list)}
        self.vocab_size = len(self.vocab)


    def build_cooccurance_matrix(self, corpus):
        """
        构建整个语料的词共现矩阵
        cooccurrences[i][j]表示词i与词j的共现值X_{ij}，此处用了距离加权（对应官方源码）
        """
        vocab_size = len(self.vocab)
        self.cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)
        for sentence in corpus:
            tokens = [self.vocab[word][0] for word in sentence]
            for center_idx, center_word in enumerate(tokens):
                start = max(center_idx - self.window, 0)
                context_tokens = tokens[start:center_idx]
                for context_idx, context_word in enumerate(context_tokens, start):
                    distance = center_idx - context_idx
                    increment = 1. / distance
                    self.cooccurrences[center_word, context_word] += increment
                    if self.symmetric:
                        self.cooccurrences[context_word, center_word] += increment


    def iter_cooccurance_matrix(self):
        """
        共现矩阵迭代器，用来迭代共现矩阵中的每个word pair
        """
        # itertools.izip作用与zip一样，只是返回一个迭代器
        for i, (row, value) in enumerate(itertools.izip(self.cooccurrences.rows, self.cooccurrences.data)):
            if self.min_count > self.id2word[i][1]:
                continue
            for value_idx, j in enumerate(row):
                if self.min_count > self.id2word[j][1]:
                    continue
                yield (i, j, value[value_idx])


    def train(self, x_max=100, power=0.75):
        """
        训练模型

        `x_max` 论文中限定的X_{ij}的上限值
        `power` 论文中的压缩比率
        """
        # 读取一遍数据（此时的data为共现矩阵），训练一次
        def train_once(data):
            data = list(data)
            self.random.shuffle(data)
            total_cost = 0.
            sample_num = len(data)
            for sample in data:
                # X_center_context 对应 X_{ij}
                i, j, X_center_context = sample
                W = self.W
                b = self.b

                # w_center 对应 w_i
                # w_context 对应 \tilde{w_j}
                # b_center 对应 b_i
                # b_context 对应 \tilde{b_j}
                w_center = W[i]
                w_context = W[j+self.vocab_size]
                b_center = b[i:i+1]
                b_context = b[j+self.vocab_size:j+self.vocab_size+1]

                # 使用adagrad迭代算法
                if self.use_adagrad:
                    # grad_square_W为W对应的累加的梯度平方
                    # grad_square_b为b对应的累加的梯度平方
                    grad_square_W = self.grad_square_W
                    grad_square_b = self.grad_square_b
                    # grad_square_w_center为w_i的梯度
                    # grad_square_w_context为\tilde{w_j}的梯度
                    # grad_square_b_center为b_i的梯度
                    # grad_square_b_context为\tilde{b_j}的梯度
                    grad_square_w_center = grad_square_W[i]
                    grad_square_w_context = grad_square_W[j+self.vocab_size]
                    grad_square_b_center = grad_square_b[i:i+1]
                    grad_square_b_context = grad_square_b[j+self.vocab_size:j+self.vocab_size+1]

                # J'=f(X_center_context)
                # J' = w_i^Tw_j + b_i + b_j - log(X_{ij})
                inner_cost = np.dot(w_center.T, w_context) + b_center + b_context - np.log(X_center_context)
                # f(X_{ij})=min(1, (X_{ij}/x_max)^power)
                weight = (1.0 * X_center_context / x_max) ** power if X_center_context < x_max else 1.
                # J=f(X_{ij})*(J')^2
                cost = weight * inner_cost ** 2
                total_cost += 0.5 * cost[0]

                # 计算梯度
                grad_w_center = weight * inner_cost * w_context
                grad_w_context = weight * inner_cost * w_center
                grad_b_center = weight * inner_cost
                grad_b_context = weight * inner_cost

                # 更新梯度
                if self.use_adagrad:
                    w_center -= grad_w_center / np.sqrt(grad_square_w_center) * self.alpha
                    w_context -= grad_w_context / np.sqrt(grad_square_w_context) * self.alpha
                    b_center -= grad_b_center / np.sqrt(grad_square_b_center) * self.alpha
                    b_context -= grad_b_context / np.sqrt(grad_square_b_context) * self.alpha
                    # 累积梯度
                    grad_square_w_center += np.square(grad_w_center)
                    grad_square_w_context += np.square(grad_w_context)
                    grad_square_b_center += np.square(grad_b_center)
                    grad_square_b_context += np.square(grad_b_context)
                else:
                    w_center -= grad_w_center * self.alpha
                    w_context -= grad_w_context * self.alpha
                    b_center -= grad_b_center * self.alpha
                    b_context -= grad_b_context * self.alpha
            return 1.0 * total_cost /sample_num

        self.reset_params()
        logger.info('start training')
        for epoch in xrange(self.epochs):
            epoch_loss = train_once(self.iter_cooccurance_matrix())
            logger.info('training epoch {}, loss {}'.format(epoch + 1, epoch_loss))

        logger.info('finished training')


    def reset_params(self):
        """
        重置参数，包括学习的权重矩阵（包含中心词向量矩阵和上下文词向量矩阵）与bias向量
        如果用adagrad迭代，还要保存累加的平方梯度
        """
        self.W = (self.random.rand(self.vocab_size * 2, self.size) - 0.5) / (self.size + 1)
        self.b = (self.random.rand(self.vocab_size * 2) - 0.5) / (self.size + 1)
        if self.use_adagrad:
            self.grad_square_W = np.ones((self.vocab_size * 2, self.size))
            self.grad_square_b = np.ones((self.vocab_size * 2))


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        找到topn个最相似的词，即与postive最相似的且与negative最不相似的词
        使用给定词的词向量余弦相似度的平均表示词类比

        Example::
          >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
          >>> model.most_similar('dog')
        """
        if isinstance(positive, basestring) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, basestring) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, basestring) else word for word in negative]

        # compute the weighted average of all words
        W = self.get_embeddings()
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if word in self.vocab:
                mean.append(weight * W[self.vocab[word][0]])
                all_words.add(self.vocab[word][0])
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = np.asarray(mean).mean(axis=0)
        mean = mean / np.linalg.norm(mean, axis=0)

        dists = np.dot(W, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.id2word[sim][0], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]


    def get_vocab(self):
        """
        获得词表字典
        """
        return self.vocab


    def get_embeddings(self):
        """
        得到大小为`vocab_size`的词向量矩阵
        """
        if self.W_norm is None:
            if self.merge_func == 'concat':
                self.W_norm = np.concatenate([self.W[:self.vocab_size], self.W[self.vocab_size:]], axis=1)
            elif self.merge_func == 'sum':
                self.W_norm = self.W[:self.vocab_size] + self.W[self.vocab_size:]
            else:
                self.W_norm = (self.W[:self.vocab_size] + self.W[self.vocab_size:]) / 2.
            self.W_norm = self.W_norm / np.linalg.norm(self.W_norm, axis=1).reshape(-1, 1)

        return self.W_norm


if __name__ == '__main__':
    test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

    glove = Glove(corpus=test_corpus, size=10, alpha=0.025, 
                 window=10, min_count=0, seed=1, iters=500, 
                 symmetric=True, sort_vocab=True, 
                 use_adagrad=True, tokenizer=None)
    similar = glove.most_similar('graph', topn=1)[0][0]
    print(similar)
    assert similar == 'trees'


