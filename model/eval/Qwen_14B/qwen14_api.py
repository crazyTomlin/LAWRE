# encoding=utf-8
import json
import os
from tqdm import tqdm
from openai import OpenAI

def calculate_similarity(q, cfact, client):
    """
    调用大模型 API 计算相似度
    """
    try:
        prompt = f"你是一位相似案例检索的法律专家，以下是两个案件的描述：\nQuery: {q}\n\n候选案件：{cfact}\n\n" \
             f"请从案情描述、关键犯罪要素等方面判断该候选案件与 Query 的相似性和相关性，相关性得分为在0和1之间的两位小数，其中0.00为完全不相似，1.00为非常相似。"\
             f"请注意，在判断相关性时，着重关注案件在关键法律要件的相似度，而不仅仅根据文本和语义的相似度判断。"\
             f"例如，两个案件都是关于寻滋挑事罪的，则两案件相关；一个案件与故意伤害有关，另一个与危险驾驶有关，则两案件不相关。"\
             f"请注意，response只需要给出一个数字，不需要输出其他字符或解释。"
             
        response = client.chat.completions.create(
            model="qwen2.5-14b-instruct",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant for legal case analysis.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        score = float(response.choices[0].message.content.strip())
        return score
    except Exception as e:
        print(f"API调用出错: {e}")
        return 0.0

def read_json_lines(file_path):
    """
    逐行读取 JSON 文件并解析为列表
    """
    cases = []
    # 读取输入文件
    with open(file_path, 'r', encoding='utf-8') as f:
        querys=f.readlines()
        for query in tqdm(querys[:]):
            cases.append(eval(query))
    return cases

def process_cases(file_path, output_path, api_key):
    """
    处理案件数据并生成相似度排序结果
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 逐行读取输入文件
    cases = read_json_lines(file_path)
    results = []
    for case in tqdm(cases, desc="Processing cases"):
        qid = case['qid']
        cid = case['cid']
        q = case['q']
        cfact = case['cfact']
        label = case['label']

        # 计算相似度
        sim = calculate_similarity(q, cfact, client)

        # 构建结果对象
        result={"qid": qid, "cid": cid, "sim": sim, "label": label}
   
        # 按行写入输出文件
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_file = "LCRCheck/cases2/keyfactor/keyfact.json"  # 输入案件文件路径
    output_file = "LCRCheck/results/test/Qwen/elam_remove_keyfact.json"  # 输出结果文件路径
    api_key = ""  # 替换为你的API Key

    process_cases(input_file, output_file, api_key)
