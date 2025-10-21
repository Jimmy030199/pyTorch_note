from torch import nn
import os


ROOT = ""
MODELS = os.path.join(ROOT,"models")
os.makedirs(MODELS,exist_ok=True)

# [dropout 解釋]
# 假設這一層的輸出是：
# [0.8, 0.5, 0.2, 0.9, 0.4]
# 若設定：
# nn.Dropout(p=0.4)
# 那在訓練時，會隨機把 40% 的神經元輸出「關掉（設為 0）」：
# → [0.8, 0, 0.2, 0.9, 0]

class IrisMLP(nn.Module):
    def __init__(self,in_dim=4,hidden1=64,hidden2=32,out_dim=3,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden2,out_dim)
        )

    def forward(self,x):
        return self.net(x)

