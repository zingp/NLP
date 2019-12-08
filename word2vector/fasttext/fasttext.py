# -*- encoding:utf8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import json
import pickle
import wget
import os
import zipfile
import heapq
import time
import itertools
import threading
import numpy.random as random
import sys
import scipy
import scipy.sparse
import math
from multiprocessing.pool import ThreadPool
from Queue import Queue
from numpy import float32 as REAL

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('fasttext')

from utils import *

"""
# Introduction

A simple code version for fasttext

# Reference
[1] https://github.com/RaRe-Technologies/gensim/blob/develop/gensim
"""


# ----- fasttext -----

# gensim里用的cython版本。
# 本来用cython版本的train_sentence会快很多，不过主要介绍原理，所以都用python介绍。
def train_sentence_sg(model, sentence, alpha, is_ft=False, work=None):
    """
    skip-gram

    `model` 训练的词向量模型
    `sentence` 一个词列表，表示一个句子
    `alpha` 学习率
    `is_ft` 表示是否是fasttext模式
    """
    total_loss = 0.
    count = 1
    # 遍历每个句子的词，该词`word`作为中心词
    for pos, word in enumerate(sentence):
        if word is None:
            # 跳过OOV的词
            continue
        # 设定一个随机的窗口大小，最终真正的单侧窗口大小为window - reduced_window
        reduced_window = model.random.randint(model.window)

        if is_ft:
            subwords_indices = [word.index]
            word2_subwords = model.ngrams_word[model.id2word[word.index]]

            for subword in word2_subwords:
                subwords_indices.append(model.ngrams[subword])

        # 遍历两侧词窗内的所有词，分别预测
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            if pos2 == pos or word2 is None:
                # 跳过OOV的词以及中心词`word`
                continue
            if is_ft:
                l1_vocab = model.syn0[subwords_indices[0]]
                l1_ngrams = np.sum(model.syn0_ngrams[subwords_indices[1:]], axis=0)
                if subwords_indices:
                    l1 = np.sum([l1_vocab, l1_ngrams], axis=0) / len(subwords_indices)
            else:
                # 得到输入词向量v_w
                # 此处本应该是`word`而不是`word2`，即求P(word2|word)的极大似然，
                # 不过这样的话，输入的投影矩阵syn0在每个词窗下只能更新一个词多次，
                # 因此此处改成求对称的P(word|word2)的极大似然，这样可以让输入的投影矩阵syn0在每个词窗下更新多个词
                # 对应fasttext则仍为word，如果是word2vec则为word2
                l1 = model.syn0[word2.index]

            # 用来存储l1的更新梯度*学习率
            neu1e = np.zeros(l1.shape, dtype=np.float32)
               
            if model.hs >= 1:
                # `outer((1 - word.code - fa), l1)`为关于l2a的梯度
                # `dot((1 - word.code - fa), l2a)`为关于l1的梯度

                # `word.point`是一个array，表示抽取出所有的当前叶子结点的结点路径
                # `l2a`是一个矩阵，shape为(codelen, layer1_size)
                l2a = model.syn1[word.point]
                # 隐层输出，`fa`的shape为(1, codelen)，每个值表示code预测为1（向树右侧走）的概率
                fa = expit(np.dot(l1, l2a.T))
                # shape为(1, codelen)
                ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                # 更新输出矩阵，shape为(codelen, layer1_size)
                model.syn1[word.point] += np.outer(ga, l1)  # learn hidden -> output

                # 更新输入矩阵
                neu1e += np.dot(ga, l2a)

                # 计算loss
                sgn = (-1.0)**word.code  # ch function, 0-> 1, 1 -> -1
                loss = sum(-np.log(expit(sgn * np.dot(l1, l2a.T))))
                total_loss += loss / len(word.code)
                count += 1
                model.running_training_loss += loss
            else:
                # `outer((model.neg_labels - fb), l1)`为关于l2a的梯度
                # `dot((model.neg_labels - fb), l2a)`为关于l1的梯度

                # 从构建的`cum_table`中找到`negative`个负样本
                neg_indices = [word.index]
                while len(neg_indices) < model.negative + 1:
                    w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                    if w != word.index:
                        neg_indices.append(w)
                # `l2b`是一个矩阵，shape为(negative+1, layer1_size)
                l2b = model.syn1neg[neg_indices]
                # shape为(1, negative+1)
                prod_term = np.dot(l1, l2b.T)
                # shape为(1, negative+1)
                fb = expit(prod_term)  # propagate hidden -> output
                # shape为(1, negative+1)
                gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
                # 更新输出矩阵，shape为(negative+1, layer1_size)
                model.syn1neg[neg_indices] += np.outer(gb, l1)  # learn hidden -> output

                # 更新输入矩阵
                neu1e += np.dot(gb, l2b)

                # 计算loss
                loss1 = -sum(np.log(expit(-1 * prod_term[1:]))) 
                loss2 = -np.log(expit(prod_term[0]))
                total_loss += (loss1 + loss2) / len(prod_term)
                count += 1
                model.running_training_loss += loss1  # for the sampled words
                model.running_training_loss += loss2  # for the output word

            # 更新输入矩阵
            if is_ft:
                # 这里的梯度需要除以len(subwords_indices)么？
                # 这块我提了issue，有人和我想的一样，是需要除的
                # 但是gensim官方说不需要除
                # 其实这个相当于是另一个学习率，对结果影响不大
                # 参考链接：https://github.com/RaRe-Technologies/gensim/issues/697
                model.syn0[subwords_indices[0]] += neu1e / len(subwords_indices)
                for i in subwords_indices[1:]:
                    model.syn0_ngrams[i] += neu1e / len(subwords_indices)
            else:
                l1 += neu1e  # save error

    # 返回句子中的非OOV的词的个数
    return len([word for word in sentence if word is not None]), 1.0 * total_loss / count


def train_sentence_cbow(model, sentence, alpha, is_ft=False, work=None, cbow_mean=True):
    """
    cbow

    `model` 训练的词向量模型
    `sentence` 一个词列表，表示一个句子
    `alpha` 学习率
    """
    total_loss = 0.
    count = 1
    for pos, word in enumerate(sentence):
        if word is None:
            continue
        reduced_window = model.random.randint(model.window)
        start = max(0, pos - model.window + reduced_window)
        window_pos = enumerate(sentence[start:(pos + model.window + 1 - reduced_window)], start)
        word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

        # 按fasttext方式
        if is_ft:
            word2_subwords = []
            # 得到上下文词的索引
            vocab_subwords_indices = []
            # 得到上下文词的subword的索引
            ngrams_subwords_indices = []

            for index in word2_indices:
                vocab_subwords_indices += [index]
                word2_subwords += model.ngrams_word[model.id2word[index]]

            for subword in word2_subwords:
                ngrams_subwords_indices.append(model.ngrams[subword])
            # 将每个上下文词的向量求和
            l1_vocab = np.sum(model.syn0[vocab_subwords_indices], axis=0)  # 1 x vector_size
            # 将每个上下文词的subword的向量求和
            l1_ngrams = np.sum(model.syn0_ngrams[ngrams_subwords_indices], axis=0)  # 1 x vector_size
            # 将两个向量相加，得到最终的上下文向量
            l1 = np.sum([l1_vocab, l1_ngrams], axis=0)
            subwords_indices = [vocab_subwords_indices] + [ngrams_subwords_indices]
            if (subwords_indices[0] or subwords_indices[1]) and cbow_mean:
                l1 /= (len(subwords_indices[0]) + len(subwords_indices[1]))
        # 按word2vec方式
        else:
            # (1, layer1_size)
            l1 = np.sum(model.syn0[word2_indices], axis=0)
            if word2_indices and cbow_mean:
                l1 /= len(word2_indices)

        neu1e = np.zeros(l1.shape, dtype=np.float32)

        if model.hs >= 1:
            # `outer((1 - word.code - fa), l1)`为关于l2a的梯度
            # `dot((1 - word.code - fa), l2a)`为关于l1的梯度

            # shape为(codelen, layer1_size)
            l2a = model.syn1[word.point]
            # shape为(1, codelen)
            fa = expit(np.dot(l1, l2a.T))
            # shape为(1, codelen)
            ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
            # 更新输出矩阵，shape为(codelen, layer1_size)
            model.syn1[word.point] += np.outer(ga, l1)  # learn hidden -> output

            # 更新输入矩阵
            neu1e += np.dot(ga, l2a)

            # 计算loss
            sgn = (-1.0)**word.code  # ch function, 0-> 1, 1 -> -1
            loss = sum(-np.log(expit(sgn * np.dot(l1, l2a.T))))
            total_loss += loss / len(word.code)
            count += 1
            model.running_training_loss += loss
        else:
            # `outer((model.neg_labels - fb), l1)`为关于l2a的梯度
            # `dot((model.neg_labels - fb), l2a)`为关于l1的梯度

            # 从构建的`cum_table`中找到`negative`个负样本
            neg_indices = [word.index]
            while len(neg_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                if w != word.index:
                    neg_indices.append(w)
            # `l2b`是一个矩阵，shape为(negative+1, layer1_size)
            l2b = model.syn1neg[neg_indices]
            # shape为(1, negative+1)
            prod_term = np.dot(l1, l2b.T)
            # shape为(1, negative+1)
            fb = expit(prod_term)  # propagate hidden -> output
            # shape为(1, negative+1)
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
            # 更新输出矩阵，shape为(negative+1, layer1_size)
            model.syn1neg[neg_indices] += np.outer(gb, l1)  # learn hidden -> output

            # 更新输入矩阵
            neu1e += np.dot(gb, l2b)

            # 计算loss
            loss1 = -sum(np.log(expit(-1 * prod_term[1:])))
            loss2 = -np.log(expit(prod_term[0]))
            total_loss += (loss1 + loss2) / len(prod_term)
            count += 1
            model.running_training_loss += loss1  # for the sampled words
            model.running_training_loss += loss2  # for the output word

        # 更新输入矩阵
        if is_ft:
            if cbow_mean and subwords_indices:
                neu1e /= (len(subwords_indices[0]) + len(subwords_indices[1]))
            for i in subwords_indices[0]:
                model.syn0[i] += (neu1e / len(subwords_indices[0]))
            for i in subwords_indices[1]:
                model.syn0_ngrams[i] += (neu1e / len(subwords_indices[1]))
        else:
            if cbow_mean and word2_indices:
                neu1e /= len(word2_indices)
            for i in word2_indices:
                model.syn0[i] += neu1e

    return len([word for word in sentence if word is not None]), 1.0 * total_loss / count


class Vocab(object):
    """
        用来存储词汇，如果用hs，则可以看成一个树结点
    """
    def __init__(self, **kwargs):
        """
        `count` 词频
        `index` 索引
        `left` 如果用hs，则表示左孩子
        `right` 如果用hs，则表示右孩子
        `code` 叶子结点的编码路径（从根结点到叶子结点的huffman编码）
        `point` 叶子结点的结点路径（从根结点到叶子结点经过的结点索引）
        """
        self.count = 0
        self.index = -1
        self.left = None
        self.right = None
        self.code = []
        self.point = []
        self.__dict__.update(kwargs)

    def __lt__(self, vocab):
        return self.count < vocab.count

    def __str__(self):
        return '<' + ', '.join([ '{}:{}'.format(
            (_, self.__dict__[_]) for _ in self.__dict__
            if not self.__dict__[_].startswith('_'))]) + '>'


# 由于在构建对象的时候直接训练模型，因此就不直接继承Word2Vec类了
class FastText(SaveAndLoad):
    def __init__(self, sentences=None, size=100, alpha=0.025, 
                 window=5, min_count=5, seed=1, iters=5, 
                 workers=1, min_alpha=0.0001, hs=0, sg=0, negative=10, 
                 sort_vocab=True, min_n=3, max_n=6, bucket=1e5):
        """
        FastText模型，其中sentences可以不给定，表示不训练模型。
        此时初始化的模型可以进行加载模型等操作

        `sentences` 输入的用于训练的句子集合，可以是迭代器，也可以是一般的list等
        `size` 词向量维度，也就是训练的时候隐变量的维度
        `window` 一句话中的当前词与上下文词的最大距离，指单侧词窗
        `alpha` 初始学习率，学习过程中随着呈线性下降趋势
        `seed` 随机数种子
        `iters` 迭代次数
        `min_count` 用于过滤的最小词频
        `workers` CPU多核的时候可以使用多线程来加快训练
        `min_alpha` 最小学习率
        `hs` 是否使用hierarchical softmax，当hs>=1时表示使用hs，否则为negative sampling
        `sg` 是否使用skip-gram，如果sg>=1表示使用sg，否则为cbow
        `negative` 表示负样本个数，用于hs<=0的情况
        `sort_vocab` 表示是否对id2word按词频从高到低排序

        以下是新加的参数
        `min_n` 表示subword的ngram最小数
        `max_n` 表示subword的ngram最大值
        `bucket` 表示共存储的subword的ngram种类数

        """
        self.window = int(window)
        self.size = int(size)
        self.min_count = int(min_count)
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = iters
        self.seed = seed
        self.random = random.RandomState(seed)
        self.workers = int(workers)
        self.hs = int(hs)
        self.sg = int(sg)
        self.negative = int(negative)
        self.sort_vocab = sort_vocab
        self.cum_table = None # negative sampling时才用到
        self.vocab = {} # 词汇表,词->Vocab类
        self.id2word = [] # id->词

        # 新加的变量
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.bucket = int(bucket)
        self.ngrams_word = {} # word->suword表
        self.ngrams = {} # subword->id, 一个id可能对应多个subword（经过hash后）
        self.num_ngram_vectors = 0 # 记录ngram总个数

        self.syn0 = [] # 即所学的词向量
        self.syn1 = [] # 用于hs
        self.syn1neg = [] # 用于负采样

        # 新加的变量
        self.syn0_ngrams = [] # (bucket, size)，表示存储的subword向量

        self.running_training_loss = 0.
        self.layer1_size = self.size
        if sentences is not None:
            self.build_vocab(sentences)
            if self.epochs is not None and self.epochs > 0:
                sentences = RepeatCorpusNTimes(sentences, self.epochs)
            self.train(sentences)

    def get_latest_training_loss(self):
        return self.running_training_loss

    def __reset_vocab(self):
        """
        重置词典和词表
        """
        self.vocab = {}
        self.word_list = []


    def build_vocab(self, sentences):
        """
        构建词典，筛除低频词，并给每个词赋予索引index
        """
        # 统计词频
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                vocab.setdefault(word, Vocab(count=0))
                vocab[word].count += 1

        # 重置词典
        self.__reset_vocab()
        # 构建词典
        for word, v in vocab.iteritems():
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.vocab[word] = v
                self.id2word.append(word)
        if self.sort_vocab:
            self.id2word = list(sorted(self.id2word, key=lambda x: self.vocab[x].count, reverse=True))
            for i, word in enumerate(self.id2word):
                self.vocab[word].index = i
        if self.hs >= 1:
            self.create_binary_tree()
        else:
            self.make_cum_table()
        self.build_ngrams()
        self.reset_weights()


    def build_ngrams(self):
        self.ngrams_word = {}
        for w in self.vocab.iterkeys():
            self.ngrams_word[w] = self.compute_ngrams(w, self.min_n, self.max_n)

        self.ngrams = {}
        all_ngrams = []
        for w, ngrams in self.ngrams_word.iteritems():
            all_ngrams += ngrams

        all_ngrams = list(set(all_ngrams))
        self.num_ngram_vectors = len(all_ngrams)
        logger.info("Total number of ngrams is %d", len(all_ngrams))

        self.hash2index = {}
        new_hash_count = 0
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = self.ft_hash(ngram) % self.bucket
            if ngram_hash in self.hash2index:
                self.ngrams[ngram] = wv.hash2index[ngram_hash]
            else:
                self.hash2index[ngram_hash] = new_hash_count
                self.ngrams[ngram] = self.hash2index[ngram_hash]
                new_hash_count = new_hash_count + 1


    def reset_weights(self):
        """
        初始化（重置）两个投影矩阵。

        syn0用于输入的词向量矩阵，
        syn1在输出为hs的时候为tree中每个内部结点
        （包括根结点共len(vocab)-1个）的向量矩阵。
        """
        self.syn0 = zeros_aligned((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0 += (self.random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size
        self.syn0_ngrams = zeros_aligned((self.bucket, self.layer1_size), dtype=REAL)
        self.syn0_ngrams += (self.random.rand(self.bucket, self.layer1_size) - 0.5) / self.layer1_size
        if self.hs >= 1:
            # syn1的索引对应的不是词，而是huffman树内部结点
            # 因此syn0与syn1不能直接结合
            self.syn1 = zeros_aligned((len(self.vocab), self.layer1_size), dtype=REAL)
        else:
            # syn0与syn1neg的索引对应的词一致，可以直接结合（如相加求平均）
            self.syn1neg = zeros_aligned((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None
        self.syn0_ngrams_norm = None


    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        构建用于negative sampling的cumulative table

        `power` 表示压缩值（论文中的值），可以有效防止负采样时候的长尾效应
        `domain` 表示采样点的范围，如果归一化到概率，则是[0, 1]，用于轮盘采样
        """
        vocab_size = len(self.id2word)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += self.vocab[self.id2word[word_index]].count ** power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += self.vocab[self.id2word[word_index]].count ** power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain


    def create_binary_tree(self):
        """
            构建huffman树，用于hierarchical softmax
            叶子结点表示词汇。
            内部结点包含该路径下的词频和与索引（该索引对应矩阵syn1）
            叶子结点包含对应词汇的词频与索引（该索引对应矩阵syn0）
            高频词出现在较短的路径上。
            叶子结点共`vocab_size`个，而内部结点（包含root结点）共`vocab_size-1`个
        """
        heap = self.vocab.values()
        heapq.heapify(heap)
        # 构建huffman二叉树，每次取词频最高的结点构成新结点
        for idx in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, 
                      index=idx + len(self.vocab), left=min1, right=min2))
        # 此时heap里只有一个结点，即根结点
        if heap:
            # max_depth记录树最大深度
            # stack中三个元素分别代表：当前结点、当前结点的编码路径、当前结点的结点路径
            max_depth, stack = 0, [(heap[0], [], [])]
            # list可以当做stack用
            while stack:
                node, codes, points = stack.pop()
                # 表示一个叶子结点
                if node.index < len(self.vocab):
                    node.code = codes
                    node.point = points
                    max_depth = max(len(codes), max_depth)
                else:
                    # 表示一个内部结点
                    # 编码路径由于是二进制，可以用unit8节省内存
                    # 这里讲内部结点的index减去一个vocab的大小，是为了保持内部结点的索引对应矩阵syn1的索引（从0开始）
                    points = np.asarray(list(points) + [node.index - len(self.vocab)], dtype=np.uint32)
                    stack.append((node.left, np.asarray(list(codes) + [0], dtype=np.uint8), points))
                    stack.append((node.right, np.asarray(list(codes) + [1], dtype=np.uint8), points))
        logger.info("built huffman tree with maximum node depth %i" % max_depth)
        self.max_depth = max_depth


    def train(self, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of utf8 strings.
        更新词向量权重
        每输入一个句子，表示一次迭代更新。

        `sentences` 输入的用于训练的句子集合
        `total_words` 总词频和
        `word_count` 词种类数
        `chunksize` 多线程的时候，每个线程一次性处理或被分配到的句子数
        """
        logger.info("training model with %i workers on %i vocabulary and %i features" % (self.workers, len(self.vocab), self.layer1_size))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        self.neg_labels = []
        if self.negative >= 1:
            # 负样本标签
            self.neg_labels = np.zeros(self.negative + 1)
            self.neg_labels[0] = 1.

        # 用来记录时间的起始与打log的判定时间，
        # `next_report`存到list是因为需要在多个线程内作为类对象被访问，`word_count`同理
        start, next_report = time.time(), [1.0]
        word_count, total_words = [word_count], total_words or sum(v.count for v in self.vocab.itervalues())
        # 考虑缓冲区
        jobs = Queue(maxsize=2 * self.workers)
        # 因为有共享变量（如`word_count`），所以加锁
        lock = threading.Lock()

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            # 多线程配给的memory
            work = zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                # 一个job就是一个chunksize条句子的数据集
                job = jobs.get()
                if job is None:  # 数据读完，退出
                    break
                # 在开始训练之前先减小训练速率
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words / self.epochs))
                # 训练参数；统计当前job训练的词数，OOV的词不算
                if self.sg >= 1:
                    func = train_sentence_sg
                else:
                    func = train_sentence_cbow
                job_words, total_loss = np.sum([func(self, sentence, alpha, True, work) for sentence in job], axis=0)
                total_loss /= len(job)
                # 请求锁，用来更新`word_count`
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    # 防止一直打log
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s, loss %.08f" %
                            (100.0 * word_count[0] / total_words / self.epochs, alpha, word_count[0] / elapsed if elapsed else 0.0, total_loss))
                        # 可以让log保持一秒及一秒以上打一次
                        next_report[0] = elapsed + 1.0

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # 可以更便捷的用ctrl+c中断程序
            thread.start()

        # 把输入的string变成Vocab类，对于OOV则用None表示，并把数据拆分成多个job，存到queue里
        no_oov = ([self.vocab.get(word, None) for word in sentence] for sentence in sentences)
        for job_no, job in enumerate(grouper(no_oov, chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        # 再补充`self.workers`个jobs，用来告知线程数据读取完成
        for _ in xrange(self.workers):
            jobs.put(None)

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]

    # ---辅助函数---
    def __getitem__(self, word):
        """
        返回word对应的词向量
        Example::
          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]
        """
        return self.word_vec(word)

    def __contains__(self, word):
        return word in self.vocab


    def similarity(self, w1, w2):
        """
        计算词w1与w2的余弦相似度
        Example::
          >>> trained_model.similarity('woman', 'man')
          0.73723527
          >>> trained_model.similarity('woman', 'woman')
          1.0
        """
        return np.dot(self.word_vec(w1, True), self.word_vec(w2, True))


    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            logger.info("precomputing L2-norms of word weight vectors")
            self.syn0norm = np.vstack(unitvec(vec) for vec in self.syn0).astype(REAL)
        if getattr(self, 'syn0_ngrams_norm', None) is None:
            logger.info("precomputing L2-norms of subword weight vectors")
            self.syn0_ngrams_norm = np.vstack(unitvec(vec) for vec in self.syn0_ngrams).astype(REAL)


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        找到topn个最相似的词，即与postive最相似的且与negative最不相似的词
        使用给定词的词向量余弦相似度的平均表示词类比

        Example::
          >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]
          >>> model.most_similar('dog')
        """
        self.init_sims()

        if isinstance(positive, basestring) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, basestring) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, basestring) else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            mean.append(weight * self.word_vec(word, True))
            if word in self.vocab:
                all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = unitvec(np.asarray(mean).mean(axis=0)).astype(REAL)

        dists = np.dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.id2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]

    def word_vec(self, word, use_norm=False):
        """
        给定一个词，返回该词的词向量
        `use_norm` 如果为True，则返回正则化的词向量
        """
        self.init_sims()
        if word in self.vocab:
            if use_norm:
                result = self.syn0norm[self.vocab[word].index]
            else:
                result = self.syn0[self.vocab[word].index]
            return result
        else:
            word_vec = np.zeros(self.syn0_ngrams.shape[1], dtype=np.float32)
            ngrams = self.compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.ngrams]
            if use_norm:
                ngram_weights = self.syn0_ngrams_norm
            else:
                ngram_weights = self.syn0_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    # 新加，直接从gensim上获取
    def compute_ngrams(self, word, min_n, max_n):
        """Returns the list of all possible ngrams for a given word.
        Parameters
        ----------
        word : str
            The word whose ngrams need to be computed
        min_n : int
            minimum character length of the ngrams
        max_n : int
            maximum character length of the ngrams
        Returns
        -------
        :obj:`list` of :obj:`str`
            List of character ngrams
        """
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        ngrams = []
        for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - ngram_length + 1):
                ngrams.append(extended_word[i:i + ngram_length])
        return ngrams

    # 新加，直接从gensim上获取
    def ft_hash(self, string):
        """Reproduces [hash method](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        used in [1]_.
        Parameter
        ---------
        string : str
            The string whose hash needs to be calculated
        Returns
        -------
        int
            The hash of the string
        """
        # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
        old_settings = np.seterr(all='ignore')
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h


if __name__ == '__main__':
    if not os.path.exists('../data'):
        os.mkdir('../data')
    # 训练语料
    text8 = Text8Corpus('../data/text8.zip', sent_len=20, sent_num=200)
    # 训练模型
    logging.info("start training model")

    # skip-gram与negative sampling
    # model = Word2Vec(sentences=text8, size=100, alpha=0.025, 
    #            window=3, min_count=5, seed=1, iters=1, 
    #            workers=1, min_alpha=0.0001, hs=0, sg=0, negative=10, 
    #            sort_vocab=True)

    # cbow与negative sampling
    # model = Word2Vec(sentences=text8, size=100, alpha=0.025, 
    #            window=3, min_count=5, seed=1, iters=1, 
    #            workers=1, min_alpha=0.0001, hs=0, sg=1, negative=10, 
    #            sort_vocab=True)

    # cbow与hierarchical softmax
    # model = Word2Vec(sentences=text8, size=100, alpha=0.025, 
    #            window=3, min_count=5, seed=1, iters=1, 
    #            workers=1, min_alpha=0.0001, hs=1, sg=0, negative=10, 
    #            sort_vocab=True)

    # skip-gram与hierarchical softmax
    # model = Word2Vec(sentences=text8, size=100, alpha=0.025, 
    #            window=3, min_count=5, seed=1, iters=1, 
    #            workers=1, min_alpha=0.0001, hs=1, sg=1, negative=10, 
    #            sort_vocab=True)

    logging.info("finished training model")

    # 简单例子测试
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
    start = time.time()
    test_corpus = [_.split() for _ in test_corpus]

    model = FastText(sentences=test_corpus, size=5, alpha=0.025, 
                 window=5, min_count=1, seed=1, iters=500, 
                 workers=2, min_alpha=0.0001, hs=1, sg=1, negative=10, 
                 sort_vocab=True, min_n=3, max_n=6, bucket=100000)
    similar = model.most_similar('graph', topn=1)[0][0]
    print(similar)
    assert similar == 'trees'
    similar = model.most_similar('tree', topn=1)[0][0]
    print(similar)
    assert similar == 'trees'
    print('elapsed time:{}s'.format(time.time() - start))

