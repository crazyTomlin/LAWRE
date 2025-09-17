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
    device = torch.device("cuda:1")
print('device=', device)

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split,train):
        if train:
            dataset = load_dataset('json',data_files='LCRCheck/cases2/keyfactor/elam_base_train.json')
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

train_dataset = Dataset('train',True)
print(len(train_dataset))


# 加载字典和分词工具
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="LCRCheck/model/bin/chinese-roberta-wwm-ext",
    cache_dir=None,
    force_download=False,
)

# 将DataLoader提供的样本打包成batch，并编码
def collate_fn(data):
    sents1 = [i[3][:509] for i in data]
    sents2 = [i[4][:3072] for i in data] 
    qids=[i[0] for i in data]
    cids=[i[1] for i in data]
    labels = [i[2] for i in data]  
    
    #编码
    encoded = tokenizer.batch_encode_plus(
        list(zip(sents1, sents2)),  
        truncation=True,
        padding=True,
        max_length=4096,
        return_tensors='pt'
    )
    
    # 将编码后的输入转换为 PyTorch 张量并移动到设备
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels,qids,cids


#数据加载器
loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=2,           #原论文32
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)


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
        self.pretrained = AutoModel.from_pretrained('LCRCheck/model/bin/Lawformer')

    def forward(self, input_ids, attention_mask, token_type_ids):
        #with torch.no_grad():   #不加这行会cuda out of memory
        out = self.pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        cls_output=out.pooler_output            
        logits = self.FFN(cls_output)  #[CLS]
        return logits

# # ranks={}
# # 训练
model = Model()
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-6,betas=(0.9, 0.999))
criterion = torch.nn.CrossEntropyLoss()

num_epochs=50
patience = 5  # 设置“早停”的容忍轮数
no_improve_count = 0  # 记录连续未改善的轮数
best_loss = float('inf')  # 初始化为无穷大
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # 累计所有 batch 的 loss
    print(epoch)
    for batch_idx, (input_ids, attention_mask, token_type_ids,
            labels,qids,cids) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        # print(out)
        # print(labels)        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 累加 batch 的 loss
        total_loss += loss.item()
    # 计算并输出每个 epoch 的平均 loss
    avg_loss = total_loss / len(loader)
    with open('LCRCheck/results/save/lawformer_elam_loss.txt','a',encoding="utf8") as f:
                 f.write(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
                 f.write("\n") 
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # 检查是否改善
    if avg_loss < best_loss:
        best_loss = avg_loss  # 更新最好的 loss
        no_improve_count = 0  # 重置计数器
    else:
        no_improve_count += 1  # 未改善的轮数 +1

    # 如果未改善轮数达到设定的阈值，则停止训练
    if no_improve_count >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

torch.save(model.state_dict(), "LCRCheck/results/save/lawformer_elam.pth")
print('已保存')

# model.eval()
# loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
#                                             batch_size=2,
#                                             collate_fn=collate_fn,
#                                             shuffle=False,
#                                             drop_last=True)


# for batch_idx,(input_ids, attention_mask, token_type_ids,
#         labels,qids,cids) in enumerate(loader_test):
#     print(batch_idx)
#     with torch.no_grad():
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     token_type_ids=token_type_ids)
#         out=out.softmax(dim=1)
#         out=out.tolist()
#         out = [round(i[1],6) for i in out]
        
#         for i in range(2):
#             result={}
#             result['qid']=qids[i]
#             result['cid']=cids[i]
#             result['sim']=out[i]
#             result['label']=labels[i].item()
#             with open('LCRCheck/results/judgment/onepeople2.json','a',encoding="utf8") as f:
#                  json.dump(result,f, ensure_ascii=False)
#                  f.write("\n")  
        


