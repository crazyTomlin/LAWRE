# encoding=utf-8
import json
import random
import os
from tqdm import tqdm
from zhipuai import ZhipuAI
# 160dd3b92a18cb9bdb4cbe7eaf1bcd75.OgA6rToWnDdPckx6
client = ZhipuAI(api_key="160dd3b92a18cb9bdb4cbe7eaf1bcd75.OgA6rToWnDdPckx6")

# 读取数据 
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        querys=f.readlines()
        data=[]
        for query in tqdm(querys[:]):
            data.append(eval(query))
    return data


# 生成某个案例的摘要
def generate_abstract(fact):
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


#将摘要转为白话版本
def generate_baihua(abstract):
    prompt=f'''
    请简要下面法律案件的摘要翻译为白话的表达方式。请注意，
    1.务必保持摘要的原意。
    2.翻译为白话后的文本需避免太多的法律专业术语，表达上尽可能贴近日常生活。
    ###法律案例的事实描述文本：{abstract}
    '''
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()


# 生成所有案例的白话摘要
def generate(query_data,file_path):
    n=0
    for query in query_data:
        n=n+1
        new_data={}
        new_data['qidx']=query['ridx']
        new_data['abstract']=generate_abstract(query['q'])
        new_data['baihua_abstract']=generate_baihua(new_data['abstract'])
        # 以追加模式打开文件 ('a' 表示 append，不会覆盖已有内容)
        with open(file_path, 'a', encoding='utf-8') as f:
            # 每次写入一个JSON对象并换行，保证符合jsonl格式
            data = json.dumps(new_data, ensure_ascii=False)
            f.write(data + '\n')
        print(n)


# 主函数
def main():
    query_path='dataset/LeCaRD-v1/data/query/query.json'
    result_path='cases/Baihua/baihua_query.json'
    # 加载query数据
    query_data = load_data(query_path)
    generate(query_data,result_path)
    #print(baihua_abstract)




if __name__ == "__main__":
    main()



