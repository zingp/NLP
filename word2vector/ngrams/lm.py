#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/ai/book-of-qna-code/tmp/lm.py
# Author: Hai Liang Wang
# Date: 2018-05-28:16:06:56
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-05-28:16:06:56"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    unicode = str

# Get ENV
ENVIRON = os.environ.copy()

import kenlm
import math
import unittest

# run testcase: python /Users/hain/ai/book-of-qna-code/tmp/lm.py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        print("加载LM模型 ...")
        model_file = os.path.join(curdir, "ngrams.arpa.gz")
        if not os.path.exists(model_file): raise BaseException("模型文件不存在!, 执行 gen_model.sh 生成模型文件。")
        self.model = kenlm.Model(model_file)

    def tearDown(self):
        pass

    def test_prob(self):
        print("kenlm: 句子出现的概率")
        print("保 险:", math.pow(10, self.model.score('保 险', bos = True, eos = True)))

    def test_perplexity(self):
        print("kenlm: 句子的困惑度")
        print("保 险:", math.pow(10, self.model.perplexity('保 险')))

def test():
    unittest.main()

if __name__ == '__main__':
    test()
