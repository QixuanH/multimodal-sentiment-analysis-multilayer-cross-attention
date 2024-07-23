from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn, optim

# 假设这是你的样本文本
sample_text = "and they knock and if you do not open the door and you wait till they bang and you do not open the door eventually it will get so frustrating that they will knock that door down"

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/deberta-v3-base/tokenizer')

inputs = tokenizer(sample_text, truncation=True, padding='max_length', max_length=32, return_tensors="pt", return_token_type_ids=False)

print(inputs)

best_model_path = '/root/FER/text_classification/best_model.pth'

class DeBERTaClassifier(nn.Module):
    def __init__(self, ):
        super(DeBERTaClassifier, self).__init__()
        self.deberta = AutoModel.from_pretrained('/root/autodl-tmp/deberta-v3-base/model')
        # 假设从DeBERTa模型中获取的隐藏层大小为768（对于base模型）
        for param in self.deberta.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
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

model.load_state_dict(torch.load(best_model_path))

model.eval()

with torch.no_grad():  # 确保在评估模式下不会计算梯度
    outputs = model(**inputs)
    print(outputs)
    # 获取预测结果，这里使用 argmax 来获取最可能的类别索引
    predicted_label = torch.argmax(outputs, dim=1)
    predicted_label = predicted_label.cpu().numpy()  # 如果在 GPU 上，需要转到 CPU

print(f"Predicted label: {predicted_label}")

