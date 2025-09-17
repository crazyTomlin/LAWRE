# encoding=utf-8
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
parser.add_argument('--elam', type=str, default='LCRCheck/dataset/要素识别/ELAM.json', help='ELAM path.')
parser.add_argument('--match', type=str, default='cases/KeyFactor/matchcase.json', help='Match path.')
args = parser.parse_args()

# 删除ELAM所有案件中的关键要素（要件事实
def remove_elam_keyfactor():
    # 打开文件
    ridx=0
    with open(args.elam, 'r',encoding="utf8") as f:
        cases = f.readlines()
    for case in tqdm(cases[:]):
        ridx+=1
        new_case={}
        new_case['qid']=ridx 
        new_case['cid']=ridx
        
        data=eval(case)
        new_case['q']=""
        new_case['cfact']=""
        
        # 删除所有的要件事实（不论该要件是否和caseB匹配)
        yjss_label_A=[]
        for i in range(0,len(data['case_A'][1])):
            if data['case_A'][1][i][1]==2: # 要件事实
                yjss_label_A.append(data['case_A'][1][i][0])                
        data['case_A'][0]=[data['case_A'][0][i] for i in range(len(data['case_A'][0])) if i not in yjss_label_A]
        for j in range(0,len(data['case_A'][0])):
            new_case['q']+=data['case_A'][0][j]
        
        yjss_label_B=[]
        for i in range(0,len(data['case_B'][1])):
            if data['case_B'][1][i][1]==2: # 要件事实
                yjss_label_B.append(data['case_B'][1][i][0])                
        data['case_B'][0]=[data['case_B'][0][i] for i in range(len(data['case_B'][0])) if i not in yjss_label_B]
        for j in range(0,len(data['case_B'][0])):
            new_case['cfact']+=data['case_B'][0][j]
               
        if data["label"]==2:  
            new_case['label']=1
        else:
            new_case['label']=0
     
        with open('LCRCheck/cases2/keyfactor/elam_remove_keyfactor.json', "a",encoding="utf8") as file:
                        json.dump(new_case, file,ensure_ascii=False)
                        file.write("\n")


# elam的baseline数据集（train
def elam_have_keyfactor():
    # 打开文件
    ridx=0
    with open(args.elam, 'r',encoding="utf8") as f:
        cases = f.readlines()
    for case in tqdm(cases[:]):
        ridx+=1
        new_case={}
        new_case['qid']=ridx 
        new_case['cid']=ridx
        
        data=eval(case)
        new_case['q']=""
        new_case['cfact']=""

        for j in range(0,len(data['case_A'][0])):
            new_case['q']+=data['case_A'][0][j]
        
        for j in range(0,len(data['case_B'][0])):
            new_case['cfact']+=data['case_B'][0][j]
               
        if data["label"]==2:   
            new_case['label']=1  #完全匹配才算匹配
        else:
            new_case['label']=0
     
        with open('LCRCheck/cases2/keyfactor/elam_base_train.json', "a",encoding="utf8") as file:
                        json.dump(new_case, file,ensure_ascii=False)
                        file.write("\n")


# 删除ELAM所有案件中的关键事实
def remove_elam_keyfact():
    # 打开文件
    ridx=0
    with open(args.elam, 'r',encoding="utf8") as f:
        cases = f.readlines()
    for case in tqdm(cases[:]):
        ridx+=1
        new_case={}
        new_case['qid']=ridx 
        new_case['cid']=ridx
        
        data=eval(case)
        new_case['q']=""
        new_case['cfact']=""
        
        # 删除所有的关键事实
        gjss_label_A=[]
        for i in range(0,len(data['case_A'][1])):
            if data['case_A'][1][i][1]==1: # 关键事实
                gjss_label_A.append(data['case_A'][1][i][0])                
        data['case_A'][0]=[data['case_A'][0][i] for i in range(len(data['case_A'][0])) if i not in gjss_label_A]
        for j in range(0,len(data['case_A'][0])):
            new_case['q']+=data['case_A'][0][j]
        
        gjss_label_B=[]
        for i in range(0,len(data['case_B'][1])):
            if data['case_B'][1][i][1]==1: # 关键事实
                gjss_label_B.append(data['case_B'][1][i][0])                
        data['case_B'][0]=[data['case_B'][0][i] for i in range(len(data['case_B'][0])) if i not in gjss_label_B]
        for j in range(0,len(data['case_B'][0])):
            new_case['cfact']+=data['case_B'][0][j]
               
        if data["label"]==2:  
            new_case['label']=1
        else:
            new_case['label']=0
     
        with open('LCRCheck/cases2/keyfactor/elam_remove_keyfact.json', "a",encoding="utf8") as file:
                        json.dump(new_case, file,ensure_ascii=False)
                        file.write("\n")

remove_elam_keyfact()

# # 删除ELAM匹配案件中的关键要素（要件事实
# def remove_keyfactor():
#     with open(args.match,'r',encoding="utf8") as f:
#         matches=f.readlines()
#         for match_case in tqdm(matches[:]):
#             new_data=eval(match_case)
#             # 删除所有的要件事实（不论该要件是否和caseB匹配)
#             yjss_label=[]
#             for i in range(0,len(new_data['case_A'][1])):
#                 if new_data['case_A'][1][i][1]==2: # 要件事实
#                     yjss_label.append(new_data['case_A'][1][i][0])
#             new_data['case_A'][0]=[new_data['case_A'][0][i] for i in range(len(new_data['case_A'][0])) if i not in yjss_label]
#             with open('cases/KeyFactor/keyfactor_elam.json', "a",encoding="utf8") as file:
#                             json.dump(new_data, file,ensure_ascii=False)
#                             file.write("\n")


# 对CAIL2019要素识别数据集做修改
def get_cail():
    with open('dataset/要素识别/data/data/divorce/data_small_selected.json','r',encoding="utf8") as f:
        cases=f.readlines()
        condition_labels=['DV1','DV3','DV10','DV11','DV12','DV14','DV19','DV20']  # 假定条件
        mottern_labels=['DV6','DV7','DV13','DV16','DV18']                         # 行为模式 
        consequence_labels=['DV2','DV4','DV5','DV8','DV9','DV15','DV17']          # 法律后果
        a,b,c=0,0,0
        print(len(cases))
        for case in tqdm(cases[:]):
            new_data=eval(case)
            #print(len(new_data))
            d,e,g=False,False,False
            for i in range(0,len(new_data)):
                one_sentence_label=new_data[i]                
                for label in one_sentence_label['labels']:
                    if label in condition_labels:
                        d=True
                    elif label in mottern_labels:
                        e=True
                    elif label in consequence_labels:
                        g=True
            if d:
                    a=a+1
            if e:
                    b=b+1
            if g:
                    c=c+1

#get_cail()
# 



             

      

