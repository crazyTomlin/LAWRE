# encoding=utf-8
import json
import os
from tqdm import tqdm
from openai import OpenAI

from ollama import chat
from ollama import ChatResponse


def chat_with_ollama(model_name = 'qwen2.5:14b', question = "你好"):     
    response: ChatResponse = chat(model=model_name, messages=[
        {
    'role': 'user',
    'content': question,
  },
])
    # print(response['message']['content'])
    # print("-"*100)
    # # or access fields directly from the response object
    print(response.message.content)
    return response.message.content


def calculate_similarity(q, cfact, client):
    """
    调用大模型 API 计算相似度
    """
    try:
        prompt = f"你是一位相似案例检索的法律专家，以下是两个案件的描述：\nQuery: {q}\n\n候选案件：{cfact}\n\n" \
             f"请从案情描述、关键犯罪要素等方面判断该候选案件与 Query 的相似性和相关性，相关性得分在0和1之间，其中0为完全不相似，1为完全相似。"\
             f"请注意，在判断相关性时，着重关注案件在关键法律要件的相似度，而不仅仅根据文本和语义的相似度判断。"\
             f"例如，两个案件都是关于寻滋挑事罪的，则两案件相关；一个案件与故意伤害有关，另一个与危险驾驶有关，则两案件不相关。"\
             f"请注意，最终相似结果用UML符号隔开，输出0-1之间的数字，如<结果>结果数字</结果>"
        print(prompt)
        response = chat_with_ollama(model_name = 'deepseek-r1:70b', question = prompt)
        print(response)
        exit()
        score = float(response)
        return score
    except Exception as e:
        print(f"API调用出错: {e}")
        return 0.0

def process_cases(file_path, output_path, api_key):
    """
    处理案件数据并生成相似度排序结果
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 读取输入文件
    with open(file_path, 'r', encoding='utf-8') as f:
        querys=f.readlines()
        cases=[]
        for query in tqdm(querys[:]):
            cases.append(eval(query))

    # 分组处理
    qid_to_cases = {}
    for case in cases:
        qid = case['qid']
        if qid not in qid_to_cases:
            qid_to_cases[qid] = []
        qid_to_cases[qid].append(case)

    results = []
    for qid, case_list in tqdm(qid_to_cases.items(), desc="Processing cases"):
        query = case_list[0]['q']  # 假设每个qid的q一致
        similarities = []

        for case in case_list:
            cid = case['cid']
            cfact = case['cfact']
            similarity = calculate_similarity(query, cfact, client)
            similarities.append((cid, similarity))

        # 根据相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        sorted_cids = [cid for cid, _ in similarities]

        results.append({"qid": qid, "sim_sort": sorted_cids})
        break

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = "LCRCheck/cases2/base/lecard_common_test.json"  # 输入案件文件路径
    output_file = "LCRCheck/results/test/Qwen/qwen14.json"  # 输出结果文件路径
    api_key = ""  # 替换为你的API Key

    process_cases(input_file, output_file, api_key)
