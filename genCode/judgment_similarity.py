# encoding=utf8

# 计算LeCaRD v1中判决结果相似度
import os
import re
import numpy as np
import json
import argparse
from tqdm import tqdm
import jieba
from sys import path

parser = argparse.ArgumentParser(description="Help info.")
#parser.add_argument('--s', type=str, default='data/others/stopword.txt', help='Stopword path.')
parser.add_argument('--w', type=str, default='cases/Exchange/exchangeCase.json', help='Write path.')
parser.add_argument('--q', type=str, default='cases/Judgment/querytt.json', help='Query path.')
parser.add_argument('--one', type=str, default='cases/Judgment/candi_one3.json', help='Candidate one path.')
parser.add_argument('--more', type=str, default='cases/Judgment/candi_more3.json', help='Candidate more path.')

args = parser.parse_args()


def remove_useless():
    with open(args.one, 'r',encoding="utf8") as f:
        ones = f.readlines()
        judgment1=[]
        for one in tqdm(ones[:]):
            judgment1.append(eval(one))
        for i in range(0,len(judgment1)):
            print(i)


 
# 提取所有案件
def extract_crime(text):
    # 定义正则表达式匹配模式
    pattern= r'(被告人|上诉人)(?P<name>\S+)犯(?P<crime>[\S、]+罪).*?判处(?P<sentence>[^，；。]+)[，；。]'
    #pattern=  r'被告人(?P<name>\S+)犯(?P<crime>[\S、]+罪).*?判处(?P<sentence>[^，；。]+)[，；。]' #目前candi正确的
    # 搜索匹配结果
    matches = re.finditer(pattern, text)
    # 提取结果
    results = []
    for match in matches:
        result = {
            'name': match.group('name'),
            'crime': [match.group('crime') or "未提及"],
            'sentence': match.group('sentence') or "未提及",           
        }
        results.append(result)
    if len(results)==0:
        pattern=  r'(被告人|上诉人)(?P<name>\S+)无罪'
        matches = re.finditer(pattern, text)
        # 提取结果
        for match in matches:
            result = {
                'name': match.group('name'),
                'crime': ["无罪"],
                'sentence': "无罪",           
            }
            results.append(result)
    #输出提取结果
    for result in results:
        print(f"被告人: {result['name']}, 罪名: {result['crime']}, 刑期: {result['sentence']}")
    return results


#提取单人多罪
def onepeople_morecrimes(pjjg):
    # 提取被告人姓名
    # name_pattern = r'(被告人|上诉人)([^\s，]+)犯'
    # name_match = re.search(name_pattern, pjjg)
    # name = name_match.group(1) if name_match else None
    name_pattern =  r'被告人([^\s，]+)犯'
    name_match = re.search(name_pattern, pjjg)
    name = name_match.group(1) if name_match else None
    if name==None:
        name_pattern =  r'上诉人([^\s，]+)犯'
        name_match = re.search(name_pattern, pjjg)
        name = name_match.group(1) if name_match else None
    # 提取罪名
    crime_pattern = r'犯([^\，；]+罪)'
    crimes = re.findall(crime_pattern, pjjg)
    # 提取最终刑期
    sentence_pattern = r'决定(合并)?执行([^\，；（）。]+)'   # 拘役？无期？死刑？
    sentence_match = re.search(sentence_pattern, pjjg)
    sentence = sentence_match.group(2) if sentence_match else None  
    #print("qidx   {}   ridx   {}     {}".format(qidx,cidx,sentence))
    # 构造结果
    result = {
        'name': name,
        'crime': crimes,
        'sentence': sentence
    }      
    return [result]


def get_data():
    all_data_list = []
    with open('cases/Judgment/querytt.json', 'r',encoding="utf8") as f:
        querys = f.readlines()
        for query in tqdm(querys[:]):
            new_data={}
            new_data['qidx']=eval(query)["ridx"]  #query id
            new_data['pjjg']=eval(query)['pjjg']  
            all_data_list.append(new_data)                    
    return all_data_list 


# 是否含判决结果
def whether_is_judgment(text):
    # 定义正则表达式匹配模式
    pattern=  r'(被告人|上诉人)(?P<name>\S+)犯(?P<crime>[\S、]+罪).*?判处(?P<sentence>[^，；。]+)[，；。]' #目前candi正确的
    # 搜索匹配结果
    matches = re.finditer(pattern, text)
    # 提取结果
    results = []
    for match in matches:
        result = {
            'name': match.group('name'),
            'crime': [match.group('crime') or "未提及"],
            'sentence': match.group('sentence') or "未提及",           
        }
        results.append(result)
    if len(results)==0:
        pattern=  r'(被告人|上诉人)(?P<name>\S+)无罪'
        matches = re.finditer(pattern, text)
        # 提取结果
        for match in matches:
            result = {
                'name': match.group('name'),
                'crime': ["无罪"],
                'sentence': "无罪",           
            }
            results.append(result)

    if len(results)>0:
        whether=True
    else:
        whether=False
    return whether
