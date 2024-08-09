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
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

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
        self.max_length = 10  # tokenizer的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle = True
        self.data_path = '/root/autodl-tmp/cmu-mosi/label.csv'
        self.num_workers = 0

def set_seed(seed_value=3407):
    """为所有可能的随机性源设置种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 实例化配置类
config = Config()
# 设置随机种子
set_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rescale_images(images):
    # 将图像数据缩放到0和1之间
    images = (images - images.min()) / (images.max() - images.min())
    return images

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_tokenizer, audio_processor, image_processor, transform=None):
        """
        Args:
            csv_file (string): 包含样本元数据的 CSV 文件路径。
            audio_dir (string): 存储音频文件的目录。
            video_dir (string): 存储视频帧的目录。
            text_tokenizer (string): 文本模型的 Tokenizer。
            audio_processor (string): 音频模型的 Processor。
            image_processor (string): 视觉模型的 Processor。
            transform (callable, optional): 对图像进行预处理的函数。
        """
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.audio_processor = WhisperProcessor.from_pretrained(audio_processor)
        self.image_processor = ViTImageProcessor.from_pretrained(image_processor)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]

        label = row['label']
        # if label == 'Negative':
        #     label = 0
        # else:
        #     label = 1

        # Load text data
        text = row['text']
        text_inputs = self.text_tokenizer(text, truncation=True, padding='max_length', max_length=24, return_tensors="pt", return_token_type_ids=False)
        text_inputs = {key: tensor.squeeze(0) for key, tensor in text_inputs.items()}

        # Load audio data
        video_id = row['video_id']
        clip_id = row['clip_id']
        audio_path = os.path.join(self.audio_dir, video_id, f"{clip_id}.wav")
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio_inputs = self.audio_processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt")
        audio_inputs = audio_inputs.input_features.squeeze(0)

        # 构建图片路径
        frame_folder = os.path.join(self.video_dir, str(video_id), str(clip_id))
        # Load image data
        frames = [os.path.join(frame_folder, f"{i}.jpg") for i in range(8)]  # Assumes 8 frames per clip
        images = [Image.open(frame).convert("RGB") for frame in frames]
        images = [self.transform(image) for image in images]
        images = torch.stack(images)
        images = rescale_images(images)  # 确保图像数据在正确的范围

        visual_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        visual_inputs = visual_inputs.pixel_values.squeeze(0)

        return text_inputs, audio_inputs, visual_inputs, torch.tensor(label).float()

# 使用示例
config = Config()

dataset = MultimodalDataset(
    csv_file=config.data_path,
    audio_dir='/root/autodl-tmp/cmu-mosi/wav',
    video_dir='/root/autodl-tmp/cmu-mosi/visual',
    text_tokenizer=config.text_model_tokenzier,
    audio_processor=config.audio_processor,
    image_processor=config.visual_processor
)

def create_dataloaders(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=8, num_workers=2):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# 使用示例
train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=config.batch_size, num_workers=config.num_workers)

# 打印第一个批次的数据
def print_first_batch(dataloader):
    batch = next(iter(dataloader))
    text_inputs, audio_inputs, visual_inputs, labels = batch
    print("Text Inputs:", text_inputs)
    print("Audio Inputs:", audio_inputs.shape)
    print("Visual Inputs:", visual_inputs.shape)
    print("Labels:", labels)

print_first_batch(train_loader)

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

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttentionLayer, self).__init__()
        self.cross_attention = CrossAttention()
        self.norm = nn.LayerNorm(feature_dim)  # 确保 feature_dim 与您模型中的特征尺寸相匹配
        self.self_attention = CrossAttention()
    def forward(self, query, key_value):
        out = self.cross_attention(query, key_value, key_value)
        key_value = key_value.squeeze(0)
        out += key_value  # 添加残差连接
        out = self.norm(out)  # 应用层归一化
        return out


class CrossModalAttentionLayer_1(nn.Module):
    def __init__(self, feature_dim):
        super(CrossModalAttentionLayer_1, self).__init__()
        self.cross_attention = CrossAttention()
        self.norm = nn.LayerNorm(feature_dim)  # 确保 feature_dim 与您模型中的特征尺寸相匹配
        self.self_attention = CrossAttention()

    def forward(self, query, key_value):
        out = self.cross_attention(query, key_value, key_value)
        key_value = key_value.squeeze(0)
        out += key_value  # 添加残差连接
        out = self.norm(out)  # 应用层归一化

        next_features = out
        next_features = next_features.unsqueeze(0)
        next_features = self.self_attention(next_features, next_features, next_features)
        
        next_features += out
        next_features = self.norm(next_features)  # 应用层归一化
        return next_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class GatedFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusionModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        combined = torch.cat((a, b), dim=1)
        gate = self.gate(combined)
        return gate * a + (1 - gate) * b

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalSentimentModel, self).__init__()
        self.text_feature_extractor = TextFeatureExtractor()
        self.audio_feature_extractor = AudioFeatureExtractor()
        self.visual_feature_extractor = VisualFeatureExtractor()

        self.pos_encoder_text = PositionalEncoding(d_model=512)  # 假设文本特征维度是512
        self.pos_encoder_audio = PositionalEncoding(d_model=512)  # 假设音频特征维度是512
        self.pos_encoder_visual = PositionalEncoding(d_model=512)  # 假设视觉特征维度是512

        self.cross_attention_text_query = CrossAttention()
        self.cross_attention_audio_query = CrossAttention()
        self.cross_attention_visual_query = CrossAttention()

        self.num_layers = 2
        self.visual_cross_attention_layers = nn.ModuleList([CrossModalAttentionLayer(512) for _ in range(self.num_layers)])
        self.audio_cross_attention_layers = nn.ModuleList([CrossModalAttentionLayer(512) for _ in range(self.num_layers)])
        self.num_layers_1 = 2
        self.cross_attention_avt_query = nn.ModuleList([CrossModalAttentionLayer(512) for _ in range(self.num_layers_1)])
        self.cross_attention_vat_query = nn.ModuleList([CrossModalAttentionLayer(512) for _ in range(self.num_layers_1)])

        # 添加自注意力层
        self.self_attention_text = CrossAttention()
        self.self_attention_audio = CrossAttention()

        # FFN layers for text and audio
        self.ffn_text1 = nn.Linear(512, 1024)
        self.ffn_text2 = nn.Linear(1024, 512)

        # Linear layers to reduce dimension from 768 to 512
        self.fc_reduce_text = nn.Linear(768, 512)

        self.gate_fusion = GatedFusionModule(512)

        self.gate_fusion1 = GatedFusionModule(512)

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

         # 应用多层交叉模态注意力
        for i in range(self.num_layers):
            audio_features = audio_features.unsqueeze(0)
            visual_features = visual_features.unsqueeze(0)
            visual_as_query = self.visual_cross_attention_layers[i](visual_features, audio_features)
            audio_as_query = self.audio_cross_attention_layers[i](audio_features, visual_features)

            # 更新输入为当前层的输出，为下一层做准备
            visual_features = visual_as_query
            audio_features = audio_as_query
            
        combined_features = (0.5 * visual_features + 0.5 * audio_features)
        # combined_features = self.gate_fusion(visual_features, audio_features)
 
        for i in range(self.num_layers):
            text_features = text_features.unsqueeze(0)
            combined_features = combined_features.unsqueeze(0)
            text_as_query = self.cross_attention_avt_query[i](text_features, combined_features)
            combine_as_query = self.cross_attention_vat_query[i](combined_features, text_features)

            # 更新输入为当前层的输出，为下一层做准备
            text_features = text_as_query
            combined_features = combine_as_query
 
        # 应用共享的FFN
        text_features = self.relu(self.ffn_text1(text_features))
        text_features = self.ffn_text2(text_features)

        combined_features = self.relu(self.ffn_text1(combined_features))
        combined_features = self.ffn_text2(combined_features)
        
        combined_features = torch.cat((text_features, combined_features), dim=1)
        
        # Fully connected layers
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)

        return output

model = MultimodalSentimentModel(num_classes=1)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

criterion = nn.MSELoss()

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for data in tqdm(train_loader, desc="Training"):
        text_inputs, audio_inputs, visual_inputs, labels = data
        text_inputs = {key: val.to(device) for key, val in text_inputs.items()}
        audio_inputs = audio_inputs.to(device)
        visual_inputs = visual_inputs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(text_inputs, audio_inputs, visual_inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        all_preds.extend(outputs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    corr, _ = pearsonr(all_labels, all_preds)
    return total_loss / len(train_loader), mae, corr

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating"):
            text_inputs, audio_inputs, visual_inputs, labels = data
            text_inputs = {key: val.to(device) for key, val in text_inputs.items()}
            audio_inputs = audio_inputs.to(device)
            visual_inputs = visual_inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(text_inputs, audio_inputs, visual_inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    corr, _ = pearsonr(all_labels, all_preds)
    return total_loss / len(val_loader), mae, corr

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            text_inputs, audio_inputs, visual_inputs, labels = data
            text_inputs = {key: val.to(device) for key, val in text_inputs.items()}
            audio_inputs = audio_inputs.to(device)
            visual_inputs = visual_inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(text_inputs, audio_inputs, visual_inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    corr, _ = pearsonr(all_labels, all_preds)
    return total_loss / len(test_loader), mae, corr

best_val_corr = -1.0  # Track the best correlation coefficient
model_path = 'best_model.pth'  # Path to save the best model

num_epochs = config.num_epochs
for epoch in range(num_epochs):
    train_loss, train_mae, train_corr = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_mae, val_corr = evaluate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train Corr: {train_corr:.4f}')
    print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f}')

    # Save the model if it has the best validation correlation coefficient so far
    if val_corr > best_val_corr:
        best_val_corr = val_corr
        torch.save(model.state_dict(), model_path)
        print(f'New best model saved with Corr: {best_val_corr:.4f}')

# Load the best model      
model.load_state_dict(torch.load(model_path))
model.to(device)

# Evaluate on test data
test_loss, test_mae, test_corr = test(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test Corr: {test_corr:.4f}')