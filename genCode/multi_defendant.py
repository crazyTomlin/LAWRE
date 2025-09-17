# encoding=utf-8
import json
import re
from tqdm import tqdm
import random


# 获取multi-defendant数据集的一些基本信息

def dataset_info():
    with open('dataset/HRN-master/dataset/dataset-v5/data_v5.03/data_valid_v5.03.jsonl', 'r',encoding="utf8") as f:
        dataset = f.readlines()
        cases=[]
        n=0
        for data in tqdm(dataset[:]):
            case=eval(data)
            case["id"]=n
            cases.append(case)
            n=n+1
        return cases

        #数据集总数  有ralation（主从犯情节）   relation总数   relation中帮助总数             
        #2350           650                       3317             3317           test       
        #2379           647                       3950             3950           valid
        #18968          5405                      27996            27996          train

        #print(eval(data)['relations'])
        #         criminal_relation.append(eval(data)['relations'])
        # print(criminal_relation)


# 每个query有50个候选candidate，其中40个多人10个单人（query全为多被告人）
def select_query_candidate(dataset):
    for i in range(0,30): 
        data={}          
        multi=random.sample(dataset,41) 
        q_num=random.randint(0,len(multi)-1) 
        data["id"]=i
        data['query']=multi[q_num]
        del multi[q_num]
        data['multi-bl']=[]
        data['multi-zc']=[]
        for one in multi:  #犯罪情节分开
            if len(one['relations'])==0:  #
                data['multi-bl'].append(one)
            else: 
                data['multi-zc'].append(one)
        with open('cases/Multi-Defendant/multi.json', 'a', encoding='utf-8') as f:
                data = json.dumps(data, ensure_ascii=False)
                f.write(data + '\n')


# 计算Jaccard相似度(罪名、法条)
def Jaccard_similarity(list1,list2):
    intersection = [x for x in list1 if x in list2]
    union = list(set(list1 + list2))
    sim=round(intersection/union,5)
    return sim


# 判决结果相似度
def judgement_similarity():
    with open('cases/Multi-Defendant/multi.json', 'r',encoding="utf8") as f:
        dataset = f.readlines()
        for data in tqdm(dataset[:]):
            case=eval(data)
            query=case['query']
            multi_binglie=case['multi-bl']
            multi_zhucong=case['multi-zc']
            for one in multi_binglie:
                can_info=one['criminals_info']
                query_info=query['criminals_info']
                for query_defendant in query_info:
                    term=query_info[query_defendant]['terms']
                    for can_defendant in can_info:
                        info=can_info[can_defendant]
                        # 罪名相似度
                        crime_similarity=Jaccard_similarity(query_info[query_defendant]['accusations'],info['accusations'])
                        # 法条相似度
                        ariticl_similarity=Jaccard_similarity(query_info[query_defendant]['laws'],info['laws'])





data=dataset_info()
select_query_candidate(data)

