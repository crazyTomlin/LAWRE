# 将LeCaRd数据集的候选案例分词、去除停词
import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
import jieba
from sys import path

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--s', type=str,
                    default='C:/Users/11649/Desktop/LCR/data/others/stopword.txt',
                    help='Stopword path.')
parser.add_argument('--can1', type=str, default='C:/Users/11649/Desktop/LCR/data/query/贪污罪.json',
                    help='Candidate path.')

args = parser.parse_args()

# 处理停词
with open(args.s, 'r', encoding="utf8") as g:
    lines = g.readlines()  # 停词
stopwords = [i.strip() for i in lines]  # 生成列表，有序
stopwords.extend(['.', '（', '）', '-'])

corpus = []
q_c = {}

with open(args.can1, "r", encoding="utf8") as g:
    files_ = json.load(g)
    corpus = []
    n = 0
    for one in files_:
        n = n + 1
        print(n)
        a = jieba.cut(one['fact'], cut_all=False)
        tem = " ".join(a).split()
        corpus.append([i for i in tem if not i in stopwords])
    with open('C:/Users/11649/Desktop/LCR/data/corpus/贪污罪/tw_corpus.json', 'w', encoding="utf8") as f:
        json.dump(corpus, f, ensure_ascii=False)
