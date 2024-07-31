import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from transformers import ViTModel
import torch.nn as nn
from tqdm import tqdm

def rescale_images(images):
    # 将图像数据缩放到0和1之间
    images = (images - images.min()) / (images.max() - images.min())
    return images

class VideoFrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.video_frame_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.processor = ViTImageProcessor.from_pretrained('/root/autodl-tmp/ViT/vit/processor')
    
    def __len__(self):
        return len(self.video_frame_data)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id = self.video_frame_data.iloc[idx, 0]
        clip_id = self.video_frame_data.iloc[idx, 1]
        label = self.video_frame_data.iloc[idx, 7]

        if label == "Positive":
            label = 1
        else: 
            label = 0

        # 构建图片路径
        frame_folder = os.path.join(self.root_dir, str(video_id), str(clip_id))
        
        frames = [os.path.join(frame_folder, f"{i}.jpg") for i in range(8)]  # Assumes 8 frames per clip

        images = [Image.open(frame).convert("RGB") for frame in frames]
        
        if self.transform:
            images = [self.transform(image) for image in images]

        # 堆叠帧
        images = torch.stack(images)
        images = rescale_images(images)  # 确保图像数据在正确的范围
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)

        features = inputs['pixel_values'].squeeze(0)  # 移除批次维度

        return features, torch.tensor(label).long()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = VideoFrameDataset(
    csv_file='/root/autodl-tmp/cmu-mosi/label.csv',
    root_dir='/root/autodl-tmp/cmu-mosi/visual',
    transform=transform
)

# 划分比例
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


# 假设你的 DataLoader 已经定义并命名为 train_loader
for images, labels in train_loader:
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    # 仅显示一个批次即可
    break

class ViT_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim, lstm_layers, bidirectional):
        super(ViT_LSTM, self).__init__()
        # 使用预训练的 ViT 模型作为特征提取器
        self.vit = ViTModel.from_pretrained('/root/autodl-tmp/ViT/vit/model')
        for param in self.vit.parameters():
            param.requires_grad = False
        # LSTM 网络
        self.lstm = nn.LSTM(
            input_size=768,  # ViT 的特征维度
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 定义全连接层
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 使用 ViT 提取特征
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)  # 调整形状以符合ViT输入需求

        outputs = self.vit(pixel_values=x)

        vit_features = outputs.last_hidden_state[:, 0, :]  # 取CLS令牌的输出

        vit_features = vit_features.view(batch_size, seq_length, -1)
        
        # LSTM 处理时间序列数据
        lstm_out, (hidden, cell) = self.lstm(vit_features)
        
        # 取 LSTM 最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]
        
        # 通过全连接层
        x = self.relu(self.fc1(last_time_step))
        x = self.fc2(x)
        
        return x

# 模型参数
num_classes = 2  # 假设是一个二分类问题
hidden_dim = 512  # LSTM 隐藏层维度
lstm_layers = 2  # LSTM 层数
bidirectional = True  # 是否使用双向 LSTM

# 实例化模型
model = ViT_LSTM(num_classes, hidden_dim, lstm_layers, bidirectional)

# 检测是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0  # 用于跟踪最佳准确率
best_model_path = 'best_model.pth'  # 最佳模型的保存路径

# 训练参数
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
    for images, labels in train_loader_tqdm:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_loader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} Training, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Testing")
    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch + 1}: Validate Accuracy: {accuracy:.2f} %')

    # 检查是否有更好的模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with accuracy: {accuracy:.2f}%")

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Final Testing"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Final accuracy of the best model on the test images: {final_accuracy:.2f} %')
