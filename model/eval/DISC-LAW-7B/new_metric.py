# encoding=utf-8
import json
import random
import os
from tqdm import tqdm
import math
import argparse


parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--model', type=str, default='lawllm', help='Model name for result file path.')
args = parser.parse_args()


    
def judgment_metric():
    # 定义需要处理的文件列表
    file_list = ['judgment.json']
    
    # 读取label
    with open('/root/yangyi/code/LCRcheck/data/cases/Judgment/one_jud_label2.json', "r",encoding="utf8") as g:
            labels=json.load(g)
    
    for filename in file_list:
        print(f"处理文件: {filename}")
        
        # 
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{filename}', 'r', encoding='utf-8') as f:
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

        # 计算 Precision@K 指标
        topK_list = [5,10,20,30]
        sp_list = []
        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sp = 0.0
                valid_queries = 0
                for key in dic.keys():   # key是qid
                    if key in label_dic:
                        # 预测列表
                        predicted_list = dic[key]
                        # 真实标签集合
                        ground_truth_set = set(label_dic[key])

                        if not ground_truth_set:
                            continue
                        
                        predicted_topK = predicted_list[:topK]
                        
                        # 计算topK中有多少是正确的
                        relevant_in_topk = len([cid for cid in predicted_topK if cid in ground_truth_set])
                        
                        sp += float(relevant_in_topk) / topK
                        valid_queries += 1

                if valid_queries > 0:
                    temK_list.append(sp/valid_queries)
                else:
                    temK_list.append(0.0)
            sp_list.append(temK_list)

        print(f"P@5: {sp_list[0][0]:.4f}")
        print(f"P@10: {sp_list[1][0]:.4f}")
        print(f"P@20: {sp_list[2][0]:.4f}")
        print(f"P@30: {sp_list[3][0]:.4f}")

        # 计算 MAP 指标
        map_list = []
        for dic in dics:
            smap = 0.0
            valid_queries = 0
            for key in dic.keys():
                if key in label_dic:
                    ground_truth_set = set(label_dic[key])
                    num_relevant = len(ground_truth_set)
                    
                    if num_relevant == 0:
                        continue

                    predicted_list = dic[key]
                    
                    ap = 0.0
                    relevant_hits = 0
                    for i, cid in enumerate(predicted_list):
                        if cid in ground_truth_set:
                            relevant_hits += 1
                            precision_at_i = relevant_hits / (i + 1)
                            ap += precision_at_i
                    
                    if num_relevant > 0:
                        smap += ap / num_relevant
                    
                    valid_queries += 1
            
            if valid_queries > 0:
                final_map = smap / valid_queries
                map_list.append(final_map)
            else:
                map_list.append(0.0)
                
        print(f"MAP: {map_list[0]:.4f}")
        print()
        





def lecard_metric():
    # 定义需要处理的文件列表
    file_list = ['lecard_common.json', 'direct_delete.json', 'fact_abstract.json', 'baihua.json']
    
    # 读取label
    with open('/root/yangyi/code/LCRcheck/data/dataset/LeCaRD-v1/data/label/label_top30_dict.json', "r",encoding="utf8") as g:
            labels=json.load(g)
    
    for filename in file_list:
        print(f"处理文件: {filename}")
        
        # 
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{filename}', 'r', encoding='utf-8') as f:
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
        
        # 计算 Precision@K 指标
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
        
        print(f"P@5: {sp_list[0][0]:.4f}")
        print(f"P@10: {sp_list[1][0]:.4f}")
        print(f"P@20: {sp_list[2][0]:.4f}")
        print(f"P@30: {sp_list[3][0]:.4f}")
        
        # 计算 MAP 指标
        map_list = []
        for dic in dics:
            smap = 0.0
            for key in dic.keys():     
                ranks = [i for i in dic[key] if str(i) in label_dic[key]] 
                rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] == 3]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] == 3])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(smap/len(dic.keys()))
        print(f"MAP: {map_list[0]:.4f}")
        print()


def confused_mtric():
    # 定义需要处理的文件列表和对应的标签文件
    file_configs = [
        {'file': 'conf_bj.json', 'label': 'bj_label.json'},
        {'file': 'conf_dq.json', 'label': 'dq_label.json'},
        {'file': 'conf_gysr.json', 'label': 'gysr_label.json'},
        {'file': 'conf_jtzs.json', 'label': 'jtzs_label.json'},
        {'file': 'conf_tw.json', 'label': 'tw_label.json'}
    ]
    
    for config in file_configs:
        print(f"处理文件: {config['file']}")
        
        # 读取对应的label文件
        with open(f'/root/yangyi/code/LCRcheck/data/cases2/confused/trainDataset/{config["label"]}', "r",encoding="utf8") as g:
                labels=json.load(g)

        # 
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{config["file"]}', 'r', encoding='utf-8') as f:
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
        
        # 计算 Precision@K 指标
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
        print(f"P@5: {sp_list[0][0]:.4f}")
        print(f"P@10: {sp_list[1][0]:.4f}")
        print(f"P@20: {sp_list[2][0]:.4f}")
        print(f"P@30: {sp_list[3][0]:.4f}")
        
        # 计算 MAP 指标
        map_list = []
        for dic in dics:
            smap = 0.0
            for key in dic.keys():     
                ranks = [i for i in dic[key] if str(i) in label_dic[key]] 
                rels = [ranks.index(i) for i in ranks if label_dic[key][str(i)] == 3]
                tem_map = 0.0
                for rel_rank in rels:
                    tem_map += float(len([j for j in ranks[:rel_rank+1] if label_dic[key][str(j)] == 3])/(rel_rank+1))
                if len(rels) > 0:
                    smap += tem_map / len(rels)
            map_list.append(smap/len(dic.keys()))
        print(f"MAP: {map_list[0]:.4f}")
        print()


# elam或cail-scm相关计算accuracy
def elam_accuracy():
    # 定义需要处理的文件列表
    file_list = ['elam_remove_keyfact.json', 'elam_remove_keyfactor.json']
    
    for filename in file_list:
        print(f"处理文件: {filename}")
        
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{filename}', 'r', encoding='utf-8') as f:
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
            
            print(f"Accuracy: {m/n:.4f}")
        print()     


def multi_metrics():
    # 定义需要处理的文件列表
    file_list = ['multi_defendant.json','new_multi_defendant.json','multi_defendant_final.json']
    
    for filename in file_list:
        print(f"处理文件: {filename}")
        
        # 读取结果文件
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{filename}', 'r', encoding='utf-8') as f:
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

        # 构建golden labels字典
        labels = {}
        for item in fact_abstract:
            qid = str(item['qid'])
            if qid not in labels:
                labels[qid] = {}
            if item['label'] == 1:  # golden case
                labels[qid][str(item['cid'])] = 3  # 设置为最高相关度等级3，与其他函数保持一致

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

        # 计算 Precision@K 指标
        topK_list = [5,10,20,30]
        sp_list = []
        for topK in topK_list:
            temK_list = []
            for dic in dics:
                sp = 0.0
                valid_queries = 0
                for key in dic.keys():   # key是qid
                    if key in label_dic and len(label_dic[key]) > 0:  # 只处理有golden的qid
                        predicted_topK = dic[key][:topK]
                        ground_truth_set = set(label_dic[key].keys())
                        relevant_in_topk = len([cid for cid in predicted_topK if str(cid) in ground_truth_set])

                        sp += float(relevant_in_topk) / topK
                        valid_queries += 1
                if valid_queries > 0:
                    temK_list.append(sp/valid_queries)
                else:
                    temK_list.append(0.0)
            sp_list.append(temK_list)
        
        print(f"P@5: {sp_list[0][0]:.4f}")
        print(f"P@10: {sp_list[1][0]:.4f}")
        print(f"P@20: {sp_list[2][0]:.4f}")
        print(f"P@30: {sp_list[3][0]:.4f}")
        
        # 计算 MAP 指标
        map_list = []
        for dic in dics:
            smap = 0.0
            valid_queries = 0
            for key in dic.keys():
                if key in label_dic and len(label_dic[key]) > 0:  # 只处理有golden的qid
                    ground_truth_set = set(label_dic[key].keys())
                    num_relevant = len(ground_truth_set)
                    if num_relevant == 0:
                        continue
                    predicted_list = dic[key]
                    ap = 0.0
                    relevant_hits = 0
                    for i, cid in enumerate(predicted_list):
                        if str(cid) in ground_truth_set:
                            relevant_hits += 1
                            precision_at_i = relevant_hits / (i + 1)
                            ap += precision_at_i
                    
                    if num_relevant > 0:
                        smap += ap / num_relevant
                    
                    valid_queries += 1
            
            if valid_queries > 0:
                final_map = smap / valid_queries
                map_list.append(final_map)
            else:
                map_list.append(0.0)
                
        print(f"MAP: {map_list[0]:.4f}")
        print()


def cail_accuracy():
    # 定义需要处理的文件列表
    file_list = ['fairness.json', 'cail_name.json', 'cail_sex.json', 'cail_race.json']
    
    for filename in file_list:
        print(f"处理文件: {filename}")
        
        with open(f'/root/yangyi/code/LCRcheck/results/test/{args.model}/{filename}', 'r', encoding='utf-8') as f:
            results=f.readlines()
            n,m=0,0
            
            for i in range(int(len(results)/2)):
                if eval(results[2*i])['label']==1:
                    n+=1
                    if eval(results[2*i])['sim']>eval(results[2*i+1])['sim']:
                        m+=1
                elif eval(results[2*i+1])['label']==1:
                    n+=1
                    if eval(results[2*i])['sim']<eval(results[2*i+1])['sim']:
                        m+=1
            
            print(f"Accuracy: {m/n:.4f}" if n > 0 else "Accuracy: 0.0000")
        print()   







if __name__ == "__main__":
    print(f"开始评估模型: {args.model}")
    print(f"运行所有评估指标")
    
    judgment_metric()
    lecard_metric()
    elam_accuracy()
    cail_accuracy()
    confused_mtric()
    multi_metrics()
    
    print("所有评估完成！")