import torch
import torch.nn as nn 
import torch.nn.functional as F

# 輸出的格式
# tensor([
#     [ 2.3, -0.8, 0.5, 1.1, -0.3, -1.7, 0.2, -2.0, 3.0, 0.8],  # 圖片1的分數
#     [-0.5, 0.1, 1.2, 0.8, 2.1, -0.2, 0.3, -1.1, 0.4, -0.7],  # 圖片2的分數
#     [ 0.2, 0.9, -1.0, 0.3, -0.6, 1.5, 0.4, -2.1, -0.8, 1.7]   # 圖片3的分數
# ])
# ---------------------------
class CNN(nn.Module):
    def __init__(self,num_classes:int=10):
        super().__init__()
        self.features =nn.Sequential(
            # nn.Conv2d(輸入通道數（灰階圖片只有 1 個 channel）, 要產生的輸出通道數, kernel_size, padding)
            nn.Conv2d(1,32,kernel_size=3,padding=1),# 輸入: 1 通道 (灰階) (1, 28, 28)-> 32 feature maps (32, 28, 28)
            nn.BatchNorm2d(32), #對剛剛產生的 32 個特徵圖 進行「批次正規化 (Batch Normalization)」
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,kernel_size=3,padding=1),# 再接一層 conv
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),# (根據影像大小)28*28 -> 14*14

            nn.Conv2d(32,64,kernel_size=3,padding=1), #32->64 feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),# 14*14 -> 7*7

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),# 展平成一維向量
            nn.Linear(64*7*7,256),# 64*7*7 = 3136 -> 256
            nn.Dropout(0.5),

            nn.Linear(256,num_classes)
        )

    # 定義資料的流動
    def forward(self,x):
        x= self.features(x)
        x= self.classifier(x)
        return x


if __name__ =="__main__":
    m=CNN()
    x= torch.randn(8,1,28,28) #(batch大小,通道數,長,寬)
    y=m(x)
    print(y.shape) 
# torch.Size([8, 10])
# | 維度位置  | 數字   | 意思                            
# | 第 1 維 | `8`  | 這個 batch 裡有 8 張圖片（batch size） 
# | 第 2 維 | `10` | 每張圖片有 10 個輸出數值（通常是分類類別數）      

    