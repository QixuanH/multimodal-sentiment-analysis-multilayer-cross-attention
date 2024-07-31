import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.batch_size = 4
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.model_name = '/root/autodl-tmp/deberta-v3-base/model'
        self.model_tokenizer = '/root/autodl-tmp/deberta-v3-base/tokenizer'
        self.max_length = 64  # tokenizer的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle = True
        self.data_path = '/root/autodl-tmp/label.csv'
        self.num_workers = 0

# 实例化配置类
config = Config()

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name=config.model_tokenizer):
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.data_frame.iloc[idx, 2]  # 获取'text'列
        label = self.data_frame.iloc[idx, 4]  # 获取'label'列
        if label == 'Positive':
            label = 0
        else:
            label = 1
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=config.max_length, return_tensors="pt", return_token_type_ids=False)
        inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}
        return inputs, torch.tensor(label, dtype=torch.long)

# 加载数据集
dataset = TextDataset(config.data_path)

# 计算每个样本的长度
lengths = []
for i in range(len(dataset)):
    text = dataset.data_frame.iloc[i, 2]
    lengths.append(len(text.split()))

# 将样本长度分组
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
length_counts = pd.cut(lengths, bins=bins).value_counts()

# 绘制柱状图
plt.figure(figsize=(10, 6))
length_counts.sort_index().plot(kind='bar')
plt.xlabel('Sample Length')
plt.ylabel('Frequency')
plt.title('Sample Length Distribution')
plt.xticks(rotation=45)
plt.show()
plt.savefig('./distribution.jpg')