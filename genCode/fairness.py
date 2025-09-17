# 公平性：替换query中的人名、性别、种族、受教育程度信息
import jieba
import jieba.posseg as pseg
import re
import json
import random
from tqdm import tqdm

# 打开文件
with open('LCRCheck/dataset/CAIL2019-SCM/test.json', 'r',encoding="utf8") as f:
    lines = f.readlines()

# 停用词
with open('LCRCheck/dataset/CAIL2019-SCM/stopword.txt', 'r',encoding="utf8") as g:
    words = g.readlines()
stopwords = [i.strip() for i in words]
stopwords.extend(['.','（','）','-','*'])

# 民族列表
races = [
    "汉族","蒙古族","回族","藏族","维吾尔族","苗族","彝族","壮族","布依族","朝鲜族",
    "满族","侗族","瑶族","白族","土家族","哈尼族","哈萨克族","傣族","黎族","傈僳族",
    "佤族","畲族","高山族","拉祜族","水族","东乡族","纳西族","景颇族","柯尔克孜族","土族",
    "达斡尔族","仫佬族","羌族","布朗族","撒拉族","毛难族","仡佬族","锡伯族","阿昌族","普米族",
    "塔吉克族","怒族","乌孜别克族","俄罗斯族","鄂温克族","崩龙族","保安族","裕固族","京族","塔塔尔族",
    "独龙族","鄂伦春族","赫哲族","门巴族","珞巴族","基诺族"
]
gender=["男","女"]

def process(case):
    # 使用jieba对查询案例A进行分词，同时获取词性
    words_with_flags = pseg.cut(case)
    # 去除人名
    #a = ''.join(word if flag != 'nr' else '*'*len(word) for word, flag in words_with_flags)
    # # 去除性别
    a=''.join(word for word, flag in words_with_flags)
    #a = a.replace("男", gender[random.randint(0,1)]).replace("女", gender[random.randint(0,1)])
    # # 去除种族
    for race in races:
         a = a.replace(race,races[random.randint(0,55)])
    return a

n=0       
#逐个处理query
for line in tqdm(lines[:]):
    n+=1
    data={}   
    #words_with_flags = pseg.cut(eval(line)['A'])
    aa=process(eval(line)['A'])
    label=(eval(line)['label'])
    bb=process(eval(line)['B'])
    cc=process(eval(line)['C'])
    data['qid']=n
    data['cid']=1
    data['q']=aa
    data['cfact']=bb
    if label=='B':
        data['label']=1
    else:
        data['label']=0
    with open('LCRCheck/cases2/fairness/cail_race.json', "a",encoding="utf8") as file:
                        json.dump(data, file,ensure_ascii=False)
                        file.write("\n")       
    data={}
    data['qid']=n
    data['cid']=2
    data['q']=aa
    data['cfact']=cc
    if label=='C':
        data['label']=1
    else:
        data['label']=0
    with open('LCRCheck/cases2/fairness/cail_race.json', "a",encoding="utf8") as file:
                        json.dump(data, file,ensure_ascii=False)
                        file.write("\n")  



    