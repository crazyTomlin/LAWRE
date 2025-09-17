# encoding=utf-8
import json
import random
import os
from tqdm import tqdm
import math
import argparse


parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--m', type=str, choices= ['NDCG', 'P', 'MAP', 'KAPPA'], default='P', help='Metric.')
parser.add_argument('--label', type=str, default='LeCaRD-main/data/label/label_top30_dict.json', help='Label file path.')
parser.add_argument('--pred', type=str, default='LeCaRD-main/data/prediction', help='Prediction dir path.')
parser.add_argument('--q', type=str, choices= ['all', 'common', 'controversial', 'test', 'test_2'], default='all', help='query set')
args = parser.parse_args()
def calculate_metrics(predicted_order, ground_truth_order):
    """
    计算P@5、P@10、P@20、P@30和MAP的函数。
    :param predicted_order: 模型的排序结果，是一个ID列表。
    :param ground_truth_order: 正确的排序结果，是一个ID列表。
    :return: 一个字典，包含P@5、P@10、P@20、P@30和MAP的值。
    """
    # 将正确的排序结果映射为索引字典，方便查找
    ground_truth_index = {item: index for index, item in enumerate(ground_truth_order)}
    
    # 初始化指标字典
    metrics = {
        "P@5": 0.0,
        "P@10": 0.0,
        "P@20": 0.0,
        "P@30": 0.0,
        "MAP": 0.0
    }
    
    # 用于存储每个位置的Precision值（用于计算MAP）
    precisions = []
    
    # 遍历模型的排序结果
    for i, item in enumerate(predicted_order):
        if item in ground_truth_index:
            # 计算当前的Precision值
            correct_items = sum(1 for j in range(i + 1) if predicted_order[j] in ground_truth_index)
            precision = correct_items / (i + 1)
            precisions.append(precision)
        
        # 计算P@5、P@10、P@20、P@30
        if i == 4:  # P@5
            metrics["P@5"] = correct_items / 5
        elif i == 9:  # P@10
            metrics["P@10"] = correct_items / 10
        elif i == 19:  # P@20
            metrics["P@20"] = correct_items / 20
        elif i == 29:  # P@30
            metrics["P@30"] = correct_items / 30
    
    # 如果没有正确预测的项目，MAP为0
    if precisions:
        metrics["MAP"] = sum(precisions) / len(precisions)
    
    return metrics

    
def judgment_metric():
    # 读取label
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/cases/Judgment/one_jud_label2.json', "r", encoding="utf8") as g:
            labels=json.load(g)

    # 
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/SAILER/judgment.json', 'r', encoding='utf-8') as f:
        results=f.readlines()
        fact_abstract=[]
        for re in tqdm(results[:]):
            one=eval(re)
            fact={}
            fact['qid']=one['qid']
            fact['cid']=one['cid']
            fact['sim']=one['sim']
            fact['label']=one['label']
            fact_abstract.append(fact)

    sorted_data = sorted(fact_abstract, key=lambda x: (x["qid"], -x["sim"]))
    qid_dict = {}
    for item in sorted_data:
        qid = str(item["qid"])
        cid = item["cid"]
        if qid not in qid_dict:
            qid_dict[qid] = []
        qid_dict[qid].append(cid)
    
    n=0
    for qid in qid_dict.keys():
        n=n+1
        predicted_order = labels[qid]
        ground_truth_order = qid_dict[qid]  # 正确的排序结果
        metrics = calculate_metrics(predicted_order, ground_truth_order)

        # 打印结果
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
# judgment_metric()


def lecard_metric():
    # 读取label
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/dataset/LeCaRD-v1/data/label/label_top30_dict.json', "r",encoding="utf8") as g:
            labels=json.load(g)

    # 
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/fact_abstract.json', 'r', encoding='utf-8') as f:
        results=f.readlines()
        fact_abstract=[]
        for re in tqdm(results[:]):
            one=eval(re)
            fact={}
            fact['qid']=one['qid']
            fact['cid']=one['cid']
            fact['sim']=one['sim']
            fact['label']=one['label']
            fact_abstract.append(fact)

    sorted_data = sorted(fact_abstract, key=lambda x: (x["qid"], -x["sim"]))
    qid_dict = {}
    for item in sorted_data:
        qid = str(item["qid"])
        cid = item["cid"]
        if qid not in qid_dict:
            qid_dict[qid] = []
        qid_dict[qid].append(cid)

    dics=[qid_dict]
    label_dic=labels
    
    if args.m == 'P': 
        topK_list = [5,10,20,30]
        sp_list = []
        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sp = 0.0
                for key in dic.keys():   # key是qid
                    ranks = [i for i in dic[key] if str(i) in list(label_dic[key].keys())]  # ranks是list，元素为dic[qid]中在top30的cid
                    sp += float(len([j for j in ranks[:topK] if label_dic[key][str(j)] == 3])/topK)
                temK_list.append(sp/len(dic.keys()))
            sp_list.append(temK_list)
        print(sp_list)
        
    elif args.m == 'MAP':
        map_list = []
        for dic in dics:
            smap = 0.0
            for key in dic.keys():     
                ranks = [i for i in dic[key] if str(i) in label_dic[key]] 
                rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] >= 2]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] == 3])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(smap/len(dic.keys()))
        print(map_list)

    # elif args.m == 'NDCG':
    #         topK_list = [10, 20, 30]
    #         ndcg_list = []
    #         for topK in topK_list:
    #             temK_list = []
    #             for dic in dics:
    #                 sndcg = 0.0
    #                 for key in dic.keys():
    #                     rawranks = []
    #                     for i in dic[key]:
    #                         if str(i) in list(label_dic[key].keys()):
    #                             rawranks.append(label_dic[key][str(i)])
    #                         else:
    #                             rawranks.append(0)
    #                     ranks = rawranks + [0]*(30-len(rawranks))
    #                     if sum(ranks) != 0:
    #                         sndcg += ndcg(ranks, list(label_dic[key].values()), topK)
    #                 temK_list.append(sndcg/len(keys))
    #             ndcg_list.append(temK_list)
    #         print(ndcg_list)

# lecard_metric()

def confused_metric():
     # 读取label
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/cases2/confused/trainDataset/gysr_label.json', "r",encoding="utf8") as g:
        labels=json.load(g)

    # 
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/conf_tw.json', 'r', encoding='utf-8') as f:
        results=f.readlines()
        fact_abstract=[]
        for re in tqdm(results[:]):
            one=eval(re)
            fact={}
            fact['qid']=one['qid']
            fact['cid']=one['cid']
            fact['sim']=one['sim']
            fact['label']=one['label']
            fact_abstract.append(fact)

    sorted_data = sorted(fact_abstract, key=lambda x: (x["qid"], -x["sim"]))
    qid_dict = {}
    for item in sorted_data:
        qid = str(item["qid"])
        cid = item["cid"]
        if qid not in qid_dict:
            qid_dict[qid] = []
        qid_dict[qid].append(cid)

    dics=[qid_dict]
    label_dic=labels
    
    if args.m == 'P': 
        topK_list = [5,10,20,30]
        sp_list = []
        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sp = 0.0
                for key in dic.keys():   # key是qid
                    ranks = [i for i in dic[key] if str(i) in list(label_dic[key].keys())]  # ranks是list，元素为dic[qid]中在top30的cid
                    sp += float(len([j for j in ranks[:topK] if label_dic[key][str(j)] == 3])/topK)
                temK_list.append(sp/len(dic.keys()))
            sp_list.append(temK_list)
        print(sp_list)
        
    elif args.m == 'MAP':
        map_list = []
        for dic in dics:
            smap = 0.0
            for key in dic.keys():     
                ranks = [i for i in dic[key] if str(i) in label_dic[key]] 
                rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] >= 2]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] == 3])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(smap/len(dic.keys()))
        print(map_list)

confused_metric()

# elam或cail-scm相关计算accuracy
def elam_accuracy():
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/elam_remove_keyfactor.json', 'r', encoding='utf-8') as f:
        results=f.readlines()
        n,m=0,0
        # for result in tqdm(results[:]):
        #     one=eval(result)
        #     if one['label']==1:
        #         n+=1
        #         if one['sim']>0.8: #测sailer和lawformer用的0.5喔
        #             m+=1
        # print(m)
        # print(n)
        # print(m/n)         
        for result in tqdm(results[:]):
            one=eval(result)
            n+=1
            if (one['label']==1 and one['sim']>0.5) or(one['label']==0 and one['sim']<0.5):
                m+=1
        print(m)
        print(n)
        print(m/n)     

# elam_accuracy()

def cail_accuracy():
    with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/fairness.json', 'r', encoding='utf-8') as f:
        results=f.readlines()
        n,m=0,0
        print(len(results))
        print(eval(results[0])['label'])
        for i in range(int(len(results)/2)):
            if eval(results[2*i])['label']==1:
                n+=1
                if eval(results[2*i])['sim']>eval(results[2*i+1])['sim']:
                    m+=1
            elif eval(results[2*i+1])['label']==1:
                n+=1
                if eval(results[2*i])['sim']<eval(results[2*i+1])['sim']:
                    m+=1
        print(m)
        print(n)
        print(m/n)   

# cail_accuracy()


# def cail_accuracy():
#     with open('LCRCheck/results/keyfactor/elam_keyfact.json', 'r', encoding='utf-8') as f:
#         results=f.readlines()
#         n,m=0,0
#         for result in tqdm(results[:]):


'''
def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value





# print('\n\n\n\n\n\n\n')
# print(qid_dict)


# # 示例数据
# qid_dict = {
#     6421: [0.009384, 0.007839],
#     6432: [0.008628, 0.005906]
# }

# labels = {
#     6421: [0.009384, 0.007839],  # 理想排序
#     6432: [0.008628, 0.005906]   # 理想排序
# }

# 计算 DCG
def compute_dcg(scores):
    dcg = 0.0
    for i, score in enumerate(scores):
        dcg += (2 ** score - 1) / math.log2(i + 2)  # i+2 因为 log2(1) = 0
    return dcg

# 计算 NDCG
def compute_ndcg(qid_dict, labels):
    ndcg_scores = {}
    for qid, pred_scores in qid_dict.items():
        if qid not in labels:
            continue
        
        # 获取理想排序
        ideal_scores = labels[qid]
        
        # 计算 DCG 和 IDCG
        dcg = compute_dcg(pred_scores)
        print(dcg)
        idcg = compute_dcg(ideal_scores)
        
        # 防止除以 0 的情况
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores[qid] = ndcg

    return ndcg_scores

# # 计算并输出 NDCG
# ndcg_results = compute_ndcg(qid_dict, labels)
# print("NDCG 值:", ndcg_results)
'''