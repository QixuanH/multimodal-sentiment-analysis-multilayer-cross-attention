from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTImageProcessor
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

def predict_single_clip(clip_path, model, processor, transform, device):
    """对单个视频剪辑的多个帧进行预测"""

    # 假设每个剪辑包含8帧
    frames = [os.path.join(clip_path, f"{i}.jpg") for i in range(8)]
    
    images = [Image.open(frame).convert("RGB") for frame in frames]
    
    # 应用转换
    if transform:
        images = [transform(image) for image in images]

    # 堆叠帧并进行标准化
    images = torch.stack(images)
    images = (images - images.min()) / (images.max() - images.min())  # rescale images
    inputs = processor(images=images, return_tensors="pt").pixel_values.to(device)

    inputs = inputs.unsqueeze(0)  # 添加一个额外的维度来表示批次大小

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_label = 'Positive' if predicted.item() == 1 else 'Negative'

    return predicted_label


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 载入模型
model_path = 'best_model.pth'  # 确保路径正确
model = ViT_LSTM(num_classes, hidden_dim, lstm_layers, bidirectional).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# 初始化 Image Processor
processor = ViTImageProcessor.from_pretrained('/root/autodl-tmp/ViT/vit/processor')

# 转换设置
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 指定单个视频剪辑的路径
single_clip_path = '/root/autodl-tmp/cmu-mosi/visual/GWuJjcEuzt8/4'  # 替换为你的具体路径

# 进行预测
predicted_label = predict_single_clip(single_clip_path, model, processor, transform, device)

# 输出剪辑路径和预测结果
print(f"Clip path: {single_clip_path}")
print(f"Predicted label: {predicted_label}")
