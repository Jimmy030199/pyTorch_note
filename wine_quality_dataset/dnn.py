from torch import nn
import os

class DNN(nn.Module):
    def __init__(self,indim=11,hidden1=64,hidden2=32,out_dim=6,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim,hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2,out_dim)
        )
        
    def forward(self,x):
        return self.net(x)