import argparse
import json
import torch
import torch.nn as nn
from datasets import load_dataset,load_from_disk
from transformers import BertTokenizer, BertModel, AdamW
from transformers import AutoModel, AutoTokenizer
import random

# 尝试Lawformer分类
device='cpu'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print('device=', device)

parser = argparse.ArgumentParser(description="Help info.")
#parser.add_argument('--s', type=str, default='data/others/stopword.txt', help='Stopword path.')
parser.add_argument('--dataset', type=str,choices= ['LeCaRD', 'CAIL', 'ELAM','direct_delete','fact-abstract','baihua',
                                                    'ELAM-keyfact','ELAM-keyfactor','fairness','fair-name',
                                                    'fair-sex','fair-race','judgment','confused','MultiDefendant'],default='confused', help='原始数据集.')
args = parser.parse_args()

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split,train):
        if not train:
            if args.dataset=='ELAM':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/keyfactor/elam_base_train.json')
            elif args.dataset=='ELAM-keyfact':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/keyfactor/elam_remove_keyfact.json') 
            elif args.dataset=='ELAM-keyfactor':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/keyfactor/elam_remove_keyfactor.json') 
            elif args.dataset=='CAIL':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/fairness/cail_test.json') 
            elif args.dataset=='fairness':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/fairness/fair_test.json') 
            elif args.dataset=='fair-name':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/fairness/cail_name.json') 
            elif args.dataset=='fair-sex':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/fairness/cail_sex.json') 
            elif args.dataset=='fair-race':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/fairness/cail_race.json') 
            elif args.dataset=='LeCaRD':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/base/lecard_common_test.json')
            elif args.dataset=='direct_delete':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/incomplete/direct_delete_test.json')
            elif args.dataset=='fact-abstract':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/incomplete/fact_abstract_test.json')
            elif args.dataset=='baihua':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/baihua/baihua_test.json') 
            elif args.dataset=='judgment':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/Judgment/one_people.json') 
            elif args.dataset=='confused':
                dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/confused/trainDataset/jtzs.json')
            elif args.dataset=='MultiDefendant':
                 dataset = load_dataset('json',data_files='/root/lxc/LCRCheckCode/LCRCheckCode/cases2/MultiDefendant/multi_defendant.json')
        self.dataset=dataset[split]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        qid = self.dataset[i]['qid']
        cid = self.dataset[i]['cid']
        label=self.dataset[i]['label']
        q = self.dataset[i]['q']
        cfact = self.dataset[i]['cfact']
        return qid,cid,label,q,cfact


test_dataset = Dataset('train',False)
print(len(test_dataset))


# 加载字典和分词工具
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="/root/lxc/LCRCheckCode/LCRCheckCode/model/bin/chinese-roberta-wwm-ext",
    cache_dir=None,
    force_download=False,
)

# 将DataLoader提供的样本打包成batch，并编码
def collate_fn(data):
    sents1 = [i[3][:100] for i in data]
    sents2 = [i[4][:409] for i in data] 
    qids=[i[0] for i in data]
    cids=[i[1] for i in data]
    labels = [i[2] for i in data]  
    
    #编码
    encoded = tokenizer.batch_encode_plus(
        list(zip(sents1, sents2)),  
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # 将编码后的输入转换为 PyTorch 张量并移动到设备
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels,qids,cids


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size=768
        #self.fc = torch.nn.Linear(768, 2)
        self.FFN = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, self.hidden_size//4),
                nn.ReLU(),
                nn.Linear(self.hidden_size//4, 2),
                )
        self.pretrained = AutoModel.from_pretrained('/root/lxc/LCRCheckCode/LCRCheckCode/DELTA_CH')

    def forward(self, input_ids, attention_mask, token_type_ids):
        #with torch.no_grad():   #不加这行会cuda out of memory
        out = self.pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        cls_output=out.pooler_output            
        logits = self.FFN(cls_output)  #[CLS]
        return logits

# # 训练
model = Model()
model.to(device)

# 加载参数
if args.dataset=='ELAM'or args.dataset=='ELAM-keyfact' or args.dataset=='ELAM-keyfactor':
    model.load_state_dict(torch.load("/root/lxc/LCRCheckCode/LCRCheckCode/results/save/Delta/delta_elam.pth"))
elif args.dataset=='CAIL' or args.dataset=='fairness' or args.dataset=='fair-name'or args.dataset=='fair-sex' or args.dataset=='fair-race':
    model.load_state_dict(torch.load("/root/lxc/LCRCheckCode/LCRCheckCode/results/save/Delta/delta_cail.pth"))
elif args.dataset=='LeCaRD' or args.dataset=='direct_delete' or  args.dataset=='fact-abstract' or args.dataset=='baihua'or args.dataset=='judgment' or args.dataset=='confused' or args.dataset=='MultiDefendant':
    model.load_state_dict(torch.load("/root/lxc/LCRCheckCode/LCRCheckCode/results/save/Delta/delta_lecard.pth"))

print('已加载')


model.eval()
loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=2,
                                            collate_fn=collate_fn,
                                            shuffle=False,
                                            drop_last=True)


for batch_idx,(input_ids, attention_mask, token_type_ids,
        labels,qids,cids) in enumerate(loader_test):
    print(batch_idx)
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out=out.softmax(dim=1)
        out=out.tolist()
        out = [round(i[1],6) for i in out]
        
        for i in range(2):
            result={}
            result['qid']=qids[i]
            result['cid']=cids[i]
            result['sim']=out[i]
            result['label']=labels[i].item()
            if args.dataset=='ELAM':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/elam.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='ELAM-keyfact':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/elam_remove_keyfact.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='ELAM-keyfactor':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/elam_remove_keyfactor.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='CAIL':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/cail_test.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")      
            elif args.dataset=='fairness':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/fairness.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")      
            elif args.dataset=='fair-name':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/cail_name.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")    
            elif args.dataset=='fair-sex':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/cail_sex.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")   
            elif args.dataset=='fair-race':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/cail_race.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='LeCaRD':        
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/lecard_common.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='direct_delete':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/direct_delete.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n")  
            elif args.dataset=='fact-abstract':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/fact_abstract.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n") 
            elif args.dataset=='baihua':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/baihua.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n") 
            elif args.dataset=='judgment':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/judgment.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n") 
            elif args.dataset=='confused':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/conf_jtzs.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n") 
            elif args.dataset=='MultiDefendant':
                with open('/root/lxc/LCRCheckCode/LCRCheckCode/results/test/Delta/multi_defendant.json','a',encoding="utf8") as f:
                    json.dump(result,f, ensure_ascii=False)
                    f.write("\n") 
        


