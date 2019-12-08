# -*- encoding:utf8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import json
import pickle
import os
import wget
import zipfile
import heapq
import time
import itertools
import sys
import scipy
import scipy.sparse
import math
from Queue import Queue


# ----- utils工具 -----

def zeros_aligned(shape, dtype, order='C', align=128):
    """Like `numpy.zeros()`, but the array will be aligned at `align` byte boundary."""
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)
    start_index = -buffer.ctypes.data % align
    return buffer[start_index : start_index + nbytes].view(dtype).reshape(shape, order=order)


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).
    >>> print list(grouper(xrange(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()

grouper = chunkize_serial


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def unitvec(vec):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or numpy array=>numpy array, scipy.sparse=>scipy.sparse).
    """
    if scipy.sparse.issparse(vec): # convert scipy.sparse to standard numpy array
        vec = vec.tocsr()
        veclen = np.sqrt(np.sum(vec.data ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        veclen = np.sqrt(np.sum(vec ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec


def cossim(vec1, vec2):
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
    vec2len = 1.0 * math.sqrt(sum(val * val for val in vec2.itervalues()))
    assert vec1len > 0.0 and vec2len > 0.0, "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1 # swap references so that we iterate over the shorter vector
    result = sum(value * vec2.get(index, 0.0) for index, value in vec1.iteritems())
    result /= vec1len * vec2len # rescale by vector lengths
    return result


class RepeatCorpusNTimes(object):
    """Wrap a `corpus` and repeat it `n` times.
    Examples
    --------
    >>> corpus = [[(1, 0.5)], []]
    >>> list(RepeatCorpusNTimes(corpus, 3)) # repeat 3 times
    [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]
    """

    def __init__(self, corpus, n):
        """
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.
        n : int
            Number of repeats for corpus.
        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in xrange(self.n):
            for document in self.corpus:
                yield document


class SaveAndLoad(object):
    """
        用于保存和读取python类对象的基类
    """
    def __init__(self):
        pass

    def save(self, fname):
        with open(fname, 'wb') as fw:
            pickle.dump(self, fw, protocol=2)
        logger.info('File saved in {}'.format(fname))

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as fr:
            return pickle.load(fr)


# ----- dataset -----

class Text8Corpus(object):
    """
    Iterate over sentences from the "text8" corpus, 
    unzipped from http://mattmahoney.net/dc/text8.zip .
    text8数据读取，用于训练 
    """
    def __init__(self, fname, sent_num=None, sent_len=1000):
        self.fname = fname
        self.sent_num = sent_num
        self.sent_len = sent_len
        if not os.path.exists(self.fname):
            wget.download('http://mattmahoney.net/dc/text8.zip', out=self.fname)
            print('Downloaded zip file `text8.zip`!')

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', self.sent_len
        idx = 0
        with zipfile.ZipFile(self.fname) as zip_text8:
            with zip_text8.open(zip_text8.getinfo('text8')) as fin:
                while True:
                    if self.sent_num is not None and idx >= self.sent_num:
                        break
                    text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                    if text == rest:  # EOF
                        sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                        if sentence:
                            idx += 1
                            yield sentence
                        break
                    last_token = text.rfind(' ')  # the last token may have been split in two... keep it for the next iteration
                    words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                    sentence.extend(words)
                    while len(sentence) >= max_sentence_length:
                        idx += 1
                        yield sentence[:max_sentence_length]
                        if self.sent_num is not None and idx >= self.sent_num:
                            break
                        sentence = sentence[max_sentence_length:]
