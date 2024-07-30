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
from sklearn.metrics import f1_score



class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 0.00001
        self.num_epochs = 3
        self.num_labels = 2
        self.model_name = '/root/autodl-tmp/whisper-base/model'
        self.model_processor = '/root/autodl-tmp/whisper-base/processor'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle = True
        self.data_path = '/root/autodl-tmp/cmu-mosi/label.csv'
        self.num_workers = 0

# 实例化配置类
config = Config()

def predict_single_sample(audio_path, model, processor, device):
    """对单个音频样本进行预测"""

    # 加载音频
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
    inputs = processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt", return_token_type_ids=False).input_features.to(device)

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_label = 'Positive' if predicted.item() == 1 else 'Negative'

    return predicted_label


class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(config.model_name)
        self.fc1 = nn.Linear(self.whisper.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.attention = nn.MultiheadAttention(embed_dim=self.whisper.config.hidden_size, num_heads=2, dropout=0.1)

        self.dropout = nn.Dropout(0.1)
        # 禁止 Whisper 模型的参数训练
        for param in self.whisper.parameters():
            param.requires_grad = False

        # for param in self.whisper.encoder.layers.parameters():
        #     param.requires_grad = True
        for param in self.whisper.encoder.layers[-1].parameters():
            param.requires_grad = True
        # for param in self.whisper.encoder.layers[-2].parameters():
        #     param.requires_grad = True
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
        outputs = outputs.permute(1, 0, 2)
        attention_output, _ = self.attention(outputs, outputs, outputs)
        attention_output = attention_output.permute(1, 0, 2)  # Transpose back to (batch, seq, feature)

        outputs = self.relu(self.fc1(attention_output[:, -1, :]))  # Only use the output of the last sequence element
        outputs = self.dropout(outputs)
        logits = self.fc2(outputs)
        return logits

# 载入最佳模型
model_path = 'best_model.pth'  # 确保路径正确
model = AudioClassificationModel(config.num_labels).to(config.device)
model.load_state_dict(torch.load(model_path, map_location=config.device))

# 初始化 Whisper 处理器
processor = WhisperProcessor.from_pretrained(config.model_processor)

# 指定单个样本的路径
single_sample_path = '/root/autodl-tmp/cmu-mosi/wav/1DmNV9C1hbY/1.wav'  # 替换为你的具体路径

# 进行预测
predicted_label = predict_single_sample(single_sample_path, model, processor, config.device)

# 输出样本路径和预测结果
print(f"Sample path: {single_sample_path}")
print(f"Predicted label: {predicted_label}")
