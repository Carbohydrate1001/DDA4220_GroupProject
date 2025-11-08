import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from classifier import AudioClassifierMLP
# --- 配置 ---
PT_PATH = '/home/nastukoi/main/test/clap_embeddings_data.pt' 

# 参数
INPUT_DIM = 512       # CLAP输出的维度
HIDDEN_DIM = 256      # 中间藏一层的神经元数量
OUTPUT_DIM = 50       # 50 个分类
NUM_EPOCHS = 50     
BATCH_SIZE = 64       
LEARNING_RATE = 0.001 


print("开始准备训练数据")

# 1. 加载数据 (这里是主要变化哦！)
print(f"正在从 {PT_PATH} 加载数据...")
loaded_data = torch.load(PT_PATH)

# 我们在字典里存的是 'embeddings' 和 'labels'
X_tensor = loaded_data['embeddings']
y_tensor = loaded_data['labels']

# 转换成 numpy 数组
X = X_tensor.cpu().numpy()
y = y_tensor.cpu().numpy()

print(f"数据加载完毕！X shape: {X.shape}, y shape: {y.shape}")


# 2. 划分训练集和验证集 (80%训练, 20%验证)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

print(f"训练集: {X_train.shape[0]} 个, 验证集: {X_val.shape[0]} 个")

# 3. 转换成PyTorch的Tensors
# 注意：Label需要是 LongTensor (长整型)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 4. 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. 准备训练
device = torch.device('cuda')
model = AudioClassifierMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"在 {device} 上开始训练")

# 训练循环
for epoch in range(NUM_EPOCHS):
    model.train() 
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播 + 优化
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证
    model.eval() # 进入评估模式
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss /= len(val_loader.dataset)
    val_accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"回合 [Epoch {epoch+1}/{NUM_EPOCHS}] - "
          f"训练损失(Train Loss): {train_loss:.4f} | "
          f"验证损失(Val Loss): {val_loss:.4f} | "
          f"验证准确率(Val Acc): {val_accuracy:.4f} ")

print("\n训练完成")

# 保存模型
torch.save(model.state_dict(), 'audio_classifier_model.pth')
print("已经把训练好的模型保存到 'audio_classifier_model.pth' ")