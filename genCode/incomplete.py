# encoding=utf-8
# 构造lecard分类数据集(普通案例的query，去除前段、中段、后段)
import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
import jieba
from sys import path
from zhipuai import ZhipuAI


parser = argparse.ArgumentParser(description="Help info.")
#parser.add_argument('--s', type=str, default='data/others/stopword.txt', help='Stopword path.')
parser.add_argument('--w', type=str, default='cases/incomplete/direct_delete.json', help='Write path.')
parser.add_argument('--q', type=str, default='dataset/LeCaRD-v1/data/query/query.json', help='Query path.')
parser.add_argument('--can', type=str, default='dataset/LeCaRD-v1/data/candidates', help='Candidate path.')
args = parser.parse_args()


# 直接去除query fact的后段
def direct_delete():
    with open(args.q, 'r',encoding="utf8") as f:
        querys = f.readlines()
        for query in tqdm(querys[:]):
            aidx=eval(query)["ridx"]  #query id
            if aidx<0 or aidx>29:
                new_data={}
                q=eval(query)['q'] 
                num=int(len(q)/3)
                q=q[:2*num]  # 直接去除后三分之一
                new_data['ridx']=aidx
                new_data['fact']=q
                new_data['crime']=eval(query)['crime']
                with open('cases/incomplete/direct_delete.json', "a",encoding="utf8") as file:
                    json.dump(new_data, file,ensure_ascii=False)
                    file.write("\n")


# 生成某个案例的摘要
def generate_abstract(fact):
    # 160dd3b92a18cb9bdb4cbe7eaf1bcd75.OgA6rToWnDdPckx6
    client = ZhipuAI(api_key="160dd3b92a18cb9bdb4cbe7eaf1bcd75.OgA6rToWnDdPckx6")
    prompt=f'''
    请简要概括下面的法律案例的事实描述的内容，为其生成摘要。请注意，
    1.务必保持案件事实的原意，保留案件的关键事实情节和法律要素，不重要的情节可以适当去除。
    2.生成的摘要字数要明显少于原文本字数，约为原文本长度的三分之一。
    ###法律案例的事实描述文本：{fact}
    '''
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()


# 获取query fact的文本摘要
def fact_abstract():
    with open('cases/Baihua/baihua_query.json', 'r', encoding='utf-8') as f:   #之前生成过了，直接提取
        querys=f.readlines()
        for query in tqdm(querys[:]):
            new_data={}
            new_data['qidx']=eval(query)['qidx']
            abstract=eval(query)['abstract']
            if abstract[:3]=="摘要：":
                abstract=abstract[3:]
            new_data['abstract']=abstract
            # 以追加模式打开文件 ('a' 表示 append，不会覆盖已有内容)
            with open('cases/incomplete/fact_abstract.json', 'a', encoding='utf-8') as f:
                # 每次写入一个JSON对象并换行，保证符合jsonl格式
                data = json.dumps(new_data, ensure_ascii=False)
                f.write(data + '\n')


def main():
    fact_abstract()


if __name__ == '__main__':
    main()