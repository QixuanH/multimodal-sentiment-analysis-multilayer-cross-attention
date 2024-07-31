import os
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, WhisperProcessor, ViTImageProcessor
from torch.utils.data import DataLoader, random_split, Dataset
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import random
from tqdm import tqdm
import torchmetrics
from sklearn.metrics import f1_score, accuracy_score

class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 0.0001
        self.num_epochs = 20
        self.num_labels = 2
        self.text_model_name = '/root/autodl-tmp/twitter-roberta-base-sentiment-latest/model'
        self.text_model_tokenzier = '/root/autodl-tmp/twitter-roberta-base-sentiment-latest/tokenizer'
        self.audio_processor = '/root/autodl-tmp/whisper-base/processor'
        self.audio_model = '/root/autodl-tmp/whisper-base/model'
        self.visual_processor = '/root/autodl-tmp/ViT/vit/processor'
        self.visual_model = '/root/autodl-tmp/ViT/vit/model'
        self.max_length = 24  # tokenizer的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle = True
        self.data_path = '/root/autodl-tmp/cmu-mosi/label.csv'
        self.num_workers = 0

# 实例化配置类
config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.deberta = AutoModel.from_pretrained(config.text_model_name)
        # 假设从DeBERTa模型中获取的隐藏层大小为768（对于base模型）
        for param in self.deberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 获取DeBERTa模型的输出
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS]标记对应的输出作为整个序列的特征
        pooled_output = outputs.last_hidden_state[:, 0]
        
        return pooled_output

class AudioFeatureExtractor(nn.Module):
    def __init__(self):
        super(AudioFeatureExtractor, self).__init__()
        self.whisper = AutoModel.from_pretrained(config.audio_model)
        # 禁止 Whisper 模型的参数训练
        for param in self.whisper.parameters():
            param.requires_grad = False

        for param in self.whisper.encoder.layers.parameters():
            param.requires_grad = True
        # for param in self.whisper.encoder.layers[-2].parameters():
        #     param.requires_grad = True

    def forward(self, input_values):
        # 特征提取
        batch_size = input_values.size(0)  # 动态获取当前批次大小
        decoder_start_token_id = self.whisper.config.decoder_start_token_id
        decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=input_values.device)
        outputs = self.whisper(input_values, decoder_input_ids=decoder_input_ids).last_hidden_state
        return outputs

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        # 使用预训练的 ViT 模型作为特征提取器
        self.vit = AutoModel.from_pretrained(config.visual_model)
        for param in self.vit.parameters():
            param.requires_grad = False
        # LSTM 网络
        self.lstm = nn.LSTM(
            input_size=768,  # ViT 的特征维度
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

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
        
        return last_time_step

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(CrossAttention, self).__init__()
        self.multihead = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

    def forward(self, query, key, value):
        # Apply attention
        attn_output, _ = self.multihead(query, key, value)
        return attn_output.squeeze(0)  # 移除序列长度维度

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalSentimentModel, self).__init__()
        self.text_feature_extractor = TextFeatureExtractor()
        self.audio_feature_extractor = AudioFeatureExtractor()
        self.visual_feature_extractor = VisualFeatureExtractor()

        self.cross_attention_text_query = CrossAttention()
        self.cross_attention_audio_query = CrossAttention()
        self.cross_attention_visual_query = CrossAttention()

        self.cross_attention_avt_query = CrossAttention()
        self.cross_attention_vat_query = CrossAttention()

        # 添加自注意力层
        self.self_attention_text = CrossAttention()
        self.self_attention_audio = CrossAttention()

        # FFN layers for text and audio
        self.ffn_text1 = nn.Linear(512, 1024)
        self.ffn_text2 = nn.Linear(1024, 512)

        # Linear layers to reduce dimension from 768 to 512
        self.fc_reduce_text = nn.Linear(768, 512)

        self.fc1 = nn.Linear(512 + 512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, text_inputs, audio_inputs, visual_inputs):
        text_features = self.text_feature_extractor(text_inputs['input_ids'], text_inputs['attention_mask'])
        audio_features = self.audio_feature_extractor(audio_inputs)
        visual_features = self.visual_feature_extractor(visual_inputs)
        text_features = self.fc_reduce_text(text_features)

        audio_features = audio_features.squeeze(1)
        text_features = text_features.unsqueeze(0)
        audio_features = audio_features.unsqueeze(0)
        visual_features = visual_features.unsqueeze(0)

        visual_as_query = self.cross_attention_visual_query(visual_features, audio_features, audio_features)
        audio_as_query = self.cross_attention_audio_query(audio_features, visual_features, visual_features)
        
        combined_features = (visual_as_query + audio_as_query) / 2

        combined_features = combined_features.unsqueeze(0)
        # print(combined_features.shape)
        # print(text_features.shape)
        text_as_query = self.cross_attention_avt_query(text_features, combined_features, combined_features)

        combine_as_query = self.cross_attention_vat_query(combined_features, text_features, text_features)

        # # 应用自注意力
        # text_as_query, _ = self.self_attention_text(text_as_query, text_as_query, text_as_query)
        # audio_as_query, _ = self.self_attention_audio(audio_as_query, audio_as_query, audio_as_query)
        # text_as_query = self.self_attention_text(text_as_query, text_as_query, text_as_query)
        # audio_as_query = self.self_attention_audio(audio_as_query, audio_as_query, audio_as_query)
        
        # 应用共享的FFN
        text_as_query = self.relu(self.ffn_text1(text_as_query))
        text_as_query = self.ffn_text2(text_as_query)

        combine_as_query = self.relu(self.ffn_text1(combine_as_query))
        combine_as_query = self.ffn_text2(combine_as_query)
        
        combined_features = torch.cat((text_as_query, combine_as_query), dim=1)

        # Fully connected layers
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

model = MultimodalSentimentModel(num_classes=2)
# Load the best model      

model_path = 'best_model_audio_text_visual.pth'
model.load_state_dict(torch.load(model_path))

input_text = "but it was really really awesome"
audio_path = "/root/autodl-tmp/cmu-mosi/wav/03bSnISJMiM/12.wav"
visual_path = "/root/autodl-tmp/cmu-mosi/visual/03bSnISJMiM/12"

def predict_single_sample(path, input_text, clip_path, model):

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_tokenzier)
    inputs_text = tokenizer(input_text, truncation=True, padding='max_length', max_length=32, return_tensors="pt", return_token_type_ids=False)
    # 初始化 Whisper 处理器
    processor = WhisperProcessor.from_pretrained(config.audio_processor)
    
    visual_processor = ViTImageProcessor.from_pretrained('/root/autodl-tmp/ViT/vit/processor')

     # 加载音频
    audio, sample_rate = torchaudio.load(path)
    # 确保音频采样率适配模型要求
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(audio)
    # 将音频转换为单声道
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # 确保音频是浮点类型
    if audio.dtype != torch.float32:
        audio = audio
    # 使用 Whisper 的 processor 处理音频
    inputs_audio = processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt", return_token_type_ids=False).input_features

    """对单个视频剪辑的多个帧进行预测"""
    # 转换设置
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 假设每个剪辑包含8帧
    frames = [os.path.join(clip_path, f"{i}.jpg") for i in range(8)]
    
    images = [Image.open(frame).convert("RGB") for frame in frames]
    
    # 应用转换
    if transform:
        images = [transform(image) for image in images]

    # 堆叠帧并进行标准化
    images = torch.stack(images)
    images = (images - images.min()) / (images.max() - images.min())  # rescale images
    inputs = visual_processor(images=images, return_tensors="pt").pixel_values
    inputs_visual = inputs.unsqueeze(0)  # 添加一个额外的维度来表示批次大小

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_text, inputs_audio, inputs_visual)
        _, predicted = torch.max(outputs, 1)
        predicted_label = 'Positive' if predicted.item() == 1 else 'Negative'

    return predicted_label

# 进行预测
predicted_label = predict_single_sample(audio_path, input_text, visual_path, model)

# 输出剪辑路径和预测结果
print(f"Audio path: {audio_path}")
print(f"Visual path: {visual_path}")
print(f"Text content: {input_text}")
print(f"Predicted label: {predicted_label}")

