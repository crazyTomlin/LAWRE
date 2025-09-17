# -*- encoding: utf-8 -*-

import jieba
import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
from gensim import corpora, models, similarities

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--s', type=str, default='C:/Users/11649/Desktop/LCR/data/others/stopword.txt',
                    help='Stopword path.')
parser.add_argument('--q', type=str, default='C:/Users/11649/Desktop/LCR/data/query/挪用公款罪.json',
                    help='Query path.')
parser.add_argument('--split', type=str, default='C:/Users/11649/Desktop/LCR/data/corpus/挪用公款罪/nygk_corpus.json',
                    help='Split corpus path.')
parser.add_argument('--w', type=str, default='C:/Users/11649/Desktop/LCR/data/prediction/tfidf_top100_nygk_nygk.json',
                    help='Write path.')

args = parser.parse_args()

with open(args.q, 'r', encoding='utf-8') as f:
    lines = json.load(f)

with open(args.split, 'r', encoding='utf-8') as f:
    raw_corpus = json.load(f)

with open(args.s, 'r', encoding='utf-8') as g:
    words = g.readlines()
stopwords = [i.strip() for i in words]
stopwords.extend(['.', '（', '）', '-'])

# 创建词典
dictionary = corpora.Dictionary(raw_corpus)
# 获取语料库
corpus = [dictionary.doc2bow(i) for i in raw_corpus]
tfidf = models.TfidfModel(corpus)
# 特征数
featureNUM = len(dictionary.token2id.keys())
# 通过TfIdf对整个语料库进行转换并将其编入索引，以准备相似性查询
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featureNUM)
# 稀疏向量.dictionary.doc2bow(doc)是把文档doc变成一个稀疏向量，[(0, 1), (1, 1)]，表明id为0,1的词汇出现了1次，至于其他词汇，没有出现。

rankdic = {}
for line in tqdm(lines[:]):
    a = jieba.cut(line['fact'], cut_all=False)
    tem = " ".join(a).split()
    q = [i for i in tem if not i in stopwords]
    new_vec = dictionary.doc2bow(q)
    # 计算向量相似度
    sim = index[tfidf[new_vec]]
    # 获取相似度最高的100个案件的索引
    sim_idx = np.array(sim).argsort()[-101:].tolist()
    # 将索引转换为候选案件的idx（从1开始编号）
    candidate_idx = [i + 1 for i in sim_idx]  # 假设原始索引从0开始，加1使其从1开始
    candidate_idx.reverse()
    rankdic[line['idx']] = candidate_idx

with open(args.w, 'w', encoding='utf-8') as f:
    json.dump(rankdic, f, ensure_ascii=False)
