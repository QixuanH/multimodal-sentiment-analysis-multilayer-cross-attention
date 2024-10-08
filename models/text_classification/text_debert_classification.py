import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 0.0001
        self.num_epochs = 20
        self.num_labels = 2
        self.model_name = '/root/autodl-tmp/twitter-roberta-base-sentiment-latest/model'
        self.model_tokenzier = '/root/autodl-tmp/twitter-roberta-base-sentiment-latest/tokenizer'
        self.max_length = 24  # tokenizer的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle = True
        self.data_path = '/root/autodl-tmp/cmu-mosi/label.csv'
        self.num_workers = 0

def set_seed(seed_value=42):
    """为所有可能的随机性源设置种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 实例化配置类
config = Config()

# 设置随机种子
set_seed(42)

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name=config.model_tokenzier):
        """
        Args:
            csv_file (string): CSV文件的路径。
            tokenizer_name (string): 使用的tokenizer的名称。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data_frame.iloc[idx, 2]  # 获取'text'列
        label = self.data_frame.iloc[idx, 7]  # 获取'label'列，你可以根据需要将label转换为torch.tensor
        if label == 'Negative':
            label = 0
        # elif label == "Positive":
            # label = 2
        else:
            label = 1
        # 使用tokenizer处理文本
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=config.max_length, return_tensors="pt",return_token_type_ids=False)
        inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}  # 移除批处理的维度

        return inputs, torch.tensor(label, dtype=torch.long)


# # 使用数据集
# dataset = TextDataset(config.data_path)
# dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)

# 从CSV文件加载整个数据集
full_dataset = TextDataset(config.data_path)

# 数据集划分为训练、验证和测试集
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 创建DataLoader
batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# 遍历数据
i = 0
for inputs, labels in enumerate(train_loader):
    print(inputs, labels)
    # 这里可以直接使用inputs和labels进行模型的训练
    if i == 1:  # 只打印两个批次的数据用于检查
        break
    i += 1

class DeBERTaClassifier(nn.Module):
    def __init__(self, ):
        super(DeBERTaClassifier, self).__init__()
        self.deberta = AutoModel.from_pretrained(config.model_name)
        # 假设从DeBERTa模型中获取的隐藏层大小为768（对于base模型）
        for param in self.deberta.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, config.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask):
        # 获取DeBERTa模型的输出
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS]标记对应的输出作为整个序列的特征
        pooled_output = outputs.last_hidden_state[:, 0]
        # 第一个全连接层
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        # 第二个全连接层
        x = self.fc2(x)
        return x


# 实例化模型和优化器
model = DeBERTaClassifier()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

criterion = nn.CrossEntropyLoss()  # 根据你的label选择适当的损失函数

# 训练函数
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_acc = 0.0
best_model_path = 'best_model.pth'  # 定义模型保存路径

# 训练循环
num_epochs = config.num_epochs
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 检查是否是最好的模型
    if val_acc > best_acc:
        best_acc = val_acc
        # 保存模型的状态字典
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved with acc: {best_acc:.4f}')

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.to(device)

# 测试模型性能
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
