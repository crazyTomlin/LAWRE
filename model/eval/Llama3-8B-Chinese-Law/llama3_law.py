# encoding=utf-8
import json
import os
import re
from tqdm import tqdm

# 导入 snippets.py 中的函数
from snippets import chat_with_ollama


def extract_score_from_response(response):
    """
    从LLM响应中提取相似度得分
    """
    response = response.replace(' ', '').replace('　', '')
    # 多种匹配模式
    patterns = [
        r'(?:相似度|相似性)(?:(?:得)?分为|是|：)\s*(\d{1,3})',  # 模式1
        r'(\d{1,3})\s*分',                                # 模式2
        r'(?:相似度|得分)[^\d]*(\d{1,3})',                 # 模式3
        r'(\d{1,3})'                                     # 模式4
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            # 过滤出0-100范围内的数字
            valid_scores = [int(score) for score in matches if score.isdigit() and 0 <= int(score) <= 100]
            if valid_scores:
                # 选择最后一个有效得分
                score = valid_scores[-1]
                # print(f"提取到相似度得分: {score}")
                return str(score)
    
    print("未找到有效的相似度得分")
    return "ERROR"


def calculate_similarity(q, cfact):
    """
    调用Ollama模型计算相似度
    """
    try:
        prompt = """#背景#
        你是一位智慧司法的相似案例检索的法律专家。
        #需求#
        1.你需要判断#输入#中的候选案件与查询案件的相似性，两者均用UML符号隔开。
        2.请从案情描述、关键犯罪要素等方面判断候选案件与查询案件的相似性和相关性，相似度得分在 0 到 100 之间。
        3.请注意，你只需要给出最终的相似度得分，即一个0-100之间精确到个位数的数字，不要给出0和100这样的极端数字。
        #输入#
        <候选案件>
        {}
        </候选案件>
        <查询案件>
        {}
        </查询案件>
        """       
        prompt = prompt.format(cfact[:3500], q[:3500])
        # 使用Ollama模型
        response = chat_with_ollama(model='mrhua/llama3-8b-chinese-lora-law_f16_q4_0', question=prompt)
        print(response)
        # 使用提取函数
        score_str = extract_score_from_response(response)
        
        if score_str != "ERROR":
            # 将0-100的得分转换为0-1的得分
            score = float(score_str) / 100.0
            print("提取到的相关性得分是:", score)
        else:
            score = 0.0
            print("提取失败，使用默认得分: 0.0")
        
        return score
    except Exception as e:
        print(f"API调用出错: {e}")
        return 0.0


def process_cases(file_path, output_path):
    """
    处理案件数据并生成相似度排序结果
    """
    
    # 读取输入文件
    with open(file_path, 'r', encoding='utf-8') as f:
        querys = f.readlines()
        cases = []
        for query in tqdm(querys[:]):
            cases.append(eval(query))

    # 清空输出文件（如果存在）
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 逐个处理每个案例
    for case in tqdm(cases, desc="Processing cases"):
        qid = case['qid']
        cid = case['cid']
        q = case['q']
        cfact = case['cfact']
        label = case['label']

        # 计算相似度
        sim = calculate_similarity(q, cfact)

        # 构建结果对象
        result = {"qid": qid, "cid": cid, "sim": sim, "label": label}

        # 按行写入输出文件
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 定义所有数据集的配置，使用更新后的路径
    datasets_config = {
        # 'ELAM': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/keyfactor/elam_base_train.json',
        #     'output': 'elam.json'
        # },
        # 'ELAM-keyfact': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/keyfactor/elam_remove_keyfact.json',
        #     'output': 'elam_remove_keyfact.json'
        # },
        # 'ELAM-keyfactor': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/keyfactor/elam_remove_keyfactor.json',
        #     'output': 'elam_remove_keyfactor.json'
        # },
        # 'CAIL': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/fairness/cail_test.json',
        #     'output': 'cail_test.json'
        # },
        # 'fairness': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/fairness/fair_test.json',
        #     'output': 'fairness.json'
        # },
        # 'fair-name': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/fairness/cail_name.json',
        #     'output': 'cail_name.json'
        # },
        # 'fair-sex': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/fairness/cail_sex.json',
        #     'output': 'cail_sex.json'
        # },
        # 'fair-race': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/fairness/cail_race.json',
        #     'output': 'cail_race.json'
        # },
        # 'LeCaRD': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/base/lecard_common_test.json',
        #     'output': 'lecard_common.json'
        # },
        # 'direct_delete': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/incomplete/direct_delete_test.json',
        #     'output': 'direct_delete.json'
        # },
        # 'fact-abstract': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/incomplete/fact_abstract_test.json',
        #     'output': 'fact_abstract.json'
        # },
        # 'baihua': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/baihua/baihua_test.json',
        #     'output': 'baihua.json'
        # },
        # 'judgment': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/Judgment/one_people.json',
        #     'output': 'judgment.json'
        # },
        # 'conf-bj': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/confused/trainDataset/bj.json',
        #     'output': 'conf_bj.json'
        # },
        # 'conf-dq': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/confused/trainDataset/dq.json',
        #     'output': 'conf_dq.json'
        # },
        # 'conf-gysr': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/confused/trainDataset/gysr.json',
        #     'output': 'conf_gysr.json'
        # },
        # 'conf-jtzs': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/confused/trainDataset/jtzs.json',
        #     'output': 'conf_jtzs.json'
        # },
        # 'conf-tw': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/confused/trainDataset/tw.json',
        #     'output': 'conf_tw.json'
        # }
        # 'multi': {
        #     'input': '/home/hz/yangyi/LCRcheck/data/cases2/MultiDefendant/multi_defendant.json',
        #     'output': 'multi_defendant.json'
        # },
        'multi_final': {
            'input': '/home/hz/yangyi/LCRcheck/data/cases2/MultiDefendant/multi_defendant_final.json',
            'output': 'multi_defendant_final.json'
        }
    }
    
    # 输出文件夹基础路径，使用新的路径
    output_base_path = "/home/hz/yangyi/LCRcheck/results/test/llama3_law"
    
    # 循环处理每个数据集
    for dataset_name, config in datasets_config.items():
        input_file = config['input']
        output_filename = config['output']
        
        print(f"\n开始处理数据集: {dataset_name}")
        print(f"输入文件: {input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在，跳过数据集 {dataset_name}: {input_file}")
            continue
        
        # 构建输出文件路径
        output_file = os.path.join(output_base_path, output_filename)
        print(f"输出文件: {output_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # 处理案件数据
            process_cases(input_file, output_file)
            print(f"✅ 数据集 {dataset_name} 处理完成")
        except Exception as e:
            print(f"❌ 处理数据集 {dataset_name} 时出错: {e}")
            continue
    
    print("\n🎉 所有数据集处理完成！")