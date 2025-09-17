# encoding=utf-8
import json
import random
import os
from tqdm import tqdm
import math
import argparse

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--label_same', type=str, default='LCRCheck/cases2/confused/data/prediction/tfidf_top100_gysr_gysr.json', help='相同罪名的label.')
parser.add_argument('--label_conf', type=str, default='LCRCheck/cases2/confused/data/prediction/tfidf_top100_gysr_gszrsw.json', help='易混淆罪名的label')
parser.add_argument('--data1', type=str, default='LCRCheck/cases2/confused/data/query/故意杀人罪.json', help='查询罪名案例.')
parser.add_argument('--data2', type=str, default='LCRCheck/cases2/confused/data/query/过失致人死亡罪.json', help='易混淆罪名案例')
parser.add_argument('--output', type=str, default='LCRCheck/cases2/confused/trainDataset/gysr.json', help='写入数据路径')

args = parser.parse_args()

# 读取label  key:str value:int
def get_label():
    # 相同罪名
    with open(args.label_same, "r",encoding="utf8") as g:
            labels1=json.load(g)
    # 易混淆罪名
    with open(args.label_conf, "r",encoding="utf8") as g:
            labels2=json.load(g)
            
    return labels1,labels2
                
def get_data():
    d1,d2={},{}
    # 查询罪名案例 qid:fact
    with open(args.data1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
        for one in data1:
            d1[one['idx']]=one['fact']
    # 易混淆罪名案例 qid：fact
    with open(args.data2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
        for one in data2:
            d2[one['idx']]=one['fact']
    # for key,value in d1.items():
    #     print(key)  # int
    #     print(value)
    return d1,d2

def genCase():
    label1,label2=get_label()
    data1,data2=get_data()

    for i in range(30):
        candi1=label1[str(i+1)][51:]
        m=0
        for one in candi1:
            m+=1
            new_data={}
            new_data['qid']=i+1
            new_data['cid']=int(one)
            new_data['q']=data1[i+1]
            new_data['cfact']=data1[int(one)]
            if m<15:
                new_data['label']=1
            else:
                new_data['label']=0
            with open(args.output, "a",encoding="utf8") as file:
                            json.dump(new_data, file,ensure_ascii=False)
                            file.write("\n")  
            #break
        candi2=label2[str(i+1)][1:51]
        for one in candi2:
            new_data={}
            new_data['qid']=i+1
            new_data['cid']=-int(one)
            new_data['q']=data1[i+1]
            new_data['cfact']=data2[int(one)]
            new_data['label']=0
            with open(args.output, "a",encoding="utf8") as file:
                            json.dump(new_data, file,ensure_ascii=False)
                            file.write("\n")  

    # new_data['qid']=i
    # new_data['cid']

def genLabel():   
     # 相同罪名
    with open(args.label_same, "r",encoding="utf8") as g:
            labels1=json.load(g)
    # 易混淆罪名
    with open(args.label_conf, "r",encoding="utf8") as g:
            labels2=json.load(g)
    n=0
    datas={}
    for one in labels1.keys():
        n+=1
        if n==50:
            break
        datas[str(one)]={}
        rank3=labels1[one][50:65]
        for i in rank3:
            datas[str(one)][i]=3
        rank2=labels2[one][:50]
        for i in rank2:
            datas[str(one)][i]=2
    with open('LCRCheck/cases2/confused/trainDataset/gysr_label.json', "a",encoding="utf8") as file:
                        json.dump(datas, file,ensure_ascii=False)
                        file.write("\n")   

genLabel()       
#genCase()