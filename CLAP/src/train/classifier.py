import torch.nn as nn
class AudioClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AudioClassifierMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x) # (输出 50 个分数)
        return x