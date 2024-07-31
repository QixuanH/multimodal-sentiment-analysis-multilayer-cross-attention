import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from transformers import WhisperProcessor, WhisperModel
import torch
from torch import nn
from tqdm import tqdm
import random
import numpy as np
from sklearn.model_selection import KFold


class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 0.00001
        self.num_epochs = 6
        self.num_labels = 2
        self.model_name = '/root/autodl-tmp/whisper-base/model'
        self.model_processor = '/root/autodl-tmp/whisper-base/processor'
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

print(config.device)

# 设置随机种子
set_seed(42)

class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, indices=None, model_name=config.model_processor):
        """
        Args:
            csv_file (string): 包含音频文件信息的 CSV 文件路径。
            audio_dir (string): 存储音频文件的根目录。
            model_name (string): 使用的 Whisper 模型名称。
        """
        self.audio_dir = audio_dir
        data = pd.read_csv(csv_file)
        if indices is not None:
            self.data = data.iloc[indices]
        else:
            self.data = data
        self.processor = WhisperProcessor.from_pretrained(model_name)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['video_id']
        clip_id = row['clip_id']
        audio_path = os.path.join(self.audio_dir, video_id, f"{clip_id}.wav")

        audio, sample_rate = torchaudio.load(audio_path)
        
        # 确保音频采样率适配模型要求
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)

        # 将音频转换为单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 确保音频是浮点类型
        if audio.dtype != torch.float32:
            audio = audio.to(torch.float32)
       
        # 使用 Whisper 的 processor 处理音频
        inputs = self.processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt", return_token_type_ids=False)
        
        input_features = inputs.input_features.squeeze(0)  # 移除额外的批次维度

        label = row['annotation']  # 获取'label'列，你可以根据需要将label转换为torch.tensor
        if label == 'Negative':
            label = 0
        # elif label == "Negative":
        #     label = 2
        else:
            label = 1
        
        return input_features, label

# 使用示例
# dataset = AudioDataset(csv_file='/path/to/your/label.csv', audio_dir='/path/to/your/wav')
# loader = DataLoader(dataset, batch_size=10, shuffle=True)
# 从CSV文件加载整个数据集
full_dataset = AudioDataset(config.data_path, audio_dir='/root/autodl-tmp/cmu-mosi/wav')

# 数据集划分为训练、验证和测试集
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 创建DataLoader
batch_size = config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 遍历数据
# i = 0
# for inputs, labels in enumerate(train_loader):
#     print(inputs, labels)
#     # 这里可以直接使用inputs和labels进行模型的训练
#     if i == 1:  # 只打印两个批次的数据用于检查
#         break
#     i += 1

class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(config.model_name)
        self.fc1 = nn.Linear(self.whisper.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        # 禁止 Whisper 模型的参数训练
        for param in self.whisper.parameters():
            param.requires_grad = False

        for param in self.whisper.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.whisper.encoder.layers[-2].parameters():
            param.requires_grad = True
        # for param in self.whisper.encoder.layers[-3].parameters():
        #     param.requires_grad = True
        # for param in self.whisper.encoder.layers[-4].parameters():
        #     param.requires_grad = True
        
        # for param in self.whisper.decoder.layers[-1].parameters():
        #     param.requires_grad = True
        # for param in self.whisper.decoder.layers[-2].parameters():
        #     param.requires_grad = True

    def forward(self, input_values):
        # 特征提取
        batch_size = input_values.size(0)  # 动态获取当前批次大小
        decoder_start_token_id = self.whisper.config.decoder_start_token_id
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=input_values.device)

        outputs = self.whisper(input_values, decoder_input_ids=decoder_input_ids).last_hidden_state
        # 通过两个全连接层

        x = self.relu(self.fc1(outputs[:, -1, :]))  # 只取最后一个时间步的输出
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{total_loss/(total+1e-5):.4f}", accuracy=f"{correct/total:.4f}")
    
    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    progress_bar.close()
    return average_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc='Validating', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=f"{total_loss/(total+1e-5):.4f}", accuracy=f"{correct/total:.4f}")
    
    average_loss = total_loss / len(dataloader)
    accuracy = correct / total
    progress_bar.close()
    return average_loss, accuracy

# 初始化模型、优化器和损失函数
model = AudioClassificationModel(config.num_labels).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

criterion = nn.CrossEntropyLoss()


best_acc = 0.0
best_model_path = 'best_model.pth'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
    train_dataset = AudioDataset(config.data_path, audio_dir='/root/autodl-tmp/cmu-mosi/wav', indices=train_idx)
    test_dataset = AudioDataset(config.data_path, audio_dir='/root/autodl-tmp/cmu-mosi/wav', indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AudioClassificationModel(config.num_labels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练和验证每一折
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc = validate(model, test_loader, criterion, config.device)

        print(f"Fold {fold+1}, Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}")

    # 保存此折的最终测试结果
    test_loss, test_acc = validate(model, test_loader, criterion, config.device)
    results.append((test_loss, test_acc))

# 输出平均测试损失和准确率
avg_test_loss = sum([result[0] for result in results]) / len(results)
avg_test_acc = sum([result[1] for result in results]) / len(results)
print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_acc:.4f}")
