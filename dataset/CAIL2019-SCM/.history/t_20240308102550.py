from datasets import load_dataset

#加载数据
#注意：如果你的网络不允许你执行这段的代码，则直接运行【从磁盘加载数据】即可，我已经给你准备了本地化的数据文件
#转载自seamew/ChnSentiCorp
dataset = load_dataset(path='lansinuote/ChnSentiCorp')