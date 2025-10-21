import torch
from torch import nn
# 「一條直線」的數學模型
class Model(nn.Module):
    """繼承自 PyTorch 的神經網路基底類別 nn.Module 的模型"""
    def __init__(self,in_dim=1,out_dim=2):
        # 呼叫父類別 建構子
        super().__init__()
        self.lin = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        return self.lin(x)

# 測試可否用
if __name__ =="__main__":
    model=Model()
    data=[
        [1.0],
        [2.0],
        [3.0],  
    ]
    x=torch.tensor(data)
    y=model(x)
    print(y)