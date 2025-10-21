import pandas as pd
import torch
from model import Model
from torch import nn,optim
import os
import matplotlib.pyplot as plt
import numpy as np

#Step1: [讀取CSV] → [資料轉Tensor] → [建立模型與設定]

# [讀取 CSV 檔案]
df = pd.read_csv("taxi_fare_training.csv") #這會把整個 CSV 檔案讀成一個 DataFrame
# print(df)
print(df["distance_km"]) #顯示某一欄（distance_km）是一個 pandas Series（一維資料）


# [資料轉Tensor]
# 讀出並轉成 numpy 陣列 
distance_km_col = df["distance_km"].to_numpy()
fare_ntd_col = df["fare_ntd"].to_numpy()

# 再轉成 torch tensor.view(-1, 1)變成二維 一筆資料只有一個特徵
# .view(a, b)	重新塑形為 (a, b)
# .view(-1, 1)	自動計算樣本數，讓每筆資料成為一行、一個特徵
# ex:
# tensor([10.2, 15.3, 7.1, 3.9])   # shape = (4,)
# 也就是形狀 [4]（一維向量）。
# .view(-1, 1) 的作用
# 複製程式碼
# .view(-1, 1)
# 會把形狀改成：

# tensor([
#  [10.2],
#  [15.3],
#  [ 7.1],
#  [ 3.9]
# ])
x_true = torch.tensor(distance_km_col,dtype=torch.float32).view(-1,1)
y_true = torch.tensor(fare_ntd_col,dtype=torch.float32).view(-1,1)

# [建立模型與設定]

# 引入 model 
net=Model()

# 設定損失函式物件（loss function object）
loss_f=nn.MSELoss() #預測值和真實值的平方誤差 再取平均值

# 設定Optimizer（優化器）根據 loss 的梯度 來更新參數
opt = optim.SGD(net.parameters(),lr=0.01) #SGD是一筆或一批更新都行

# Step2 [訓練(50回合)]
loss_hist=[]
for epoch in range(50):
    y_hat=net(x_true)

    # loss 是用 125 筆的平均誤差 算出來的
    loss=loss_f(y_true,y_hat)
    opt.zero_grad()
    # 算梯度: 對 loss 的偏微分（∂loss/∂參數）
    loss.backward()
    # 更新參數
    opt.step()

    # .item() → 把 Tensor 轉成純數值（不帶梯度）
    #ex: tensor(0.1245, grad_fn=<MseLossBackward0>)→ 0.1245
    loss_hist.append(float(loss.item()))

    print(f"epoch:{epoch + 1},loss:{loss.item()}")

# Step3 [Loss曲線輸出]
OUT_DIR ="results"
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)),loss_hist,marker='o')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')

plt.tight_layout()

loss_path=os.path.join(OUT_DIR,'loss.png')
plt.savefig(loss_path,dpi=150)

# 如果你不關閉圖表，下一次畫圖會疊在舊圖上，或記憶體一直累積。
plt.close()


#Step4 [模型推論與評估]:畫出模型預測結果與原始資料的對照圖。

# 切到推論模式
net.eval()

with torch.no_grad(): # 關閉梯度追蹤
    y_hat = net(x_true) # 預測

    #存csv用 
    arr=np.hstack([
        x_true.detach().cpu().numpy(),
        y_true.detach().cpu().numpy(),
        y_hat.detach().cpu().numpy(),
    ])
    # 詳細說明：
    # .detach()：切斷 tensor 與計算圖（不再追蹤梯度）
    # .cpu()：確保資料在 CPU 上（GPU tensor 不能直接轉 numpy）
    # .numpy()：轉成 NumPy 格式，方便存成檔案
    # np.hstack([...])：水平合併三個陣列 → [x, y_true, y_pred]
    # np.savetxt()：把結果存成 CSV

    pred_csv_path =os.path.join(OUT_DIR,'predictions.csv')
    np.savetxt(pred_csv_path,arr,delimiter='-',comments='')

    # 目的:去拿到x_true的序號大小排列(x軸大小一定要按照順序排好 否則"劃線時"會亂畫一通)
    # .squeeze(1) 的意思是：
    # 把張量（Tensor）中第 1 維度如果是 1，就把那個維度「擠掉」（移除掉）。
    # argsort 不是用來「排序資料本身」，
    # 而是用來「回傳排序後的索引 (index) 順序」
    idx = x_true.squeeze(1).argsort()

    x_sorted=x_true[idx]
    yhat_sorted=y_hat[idx]
    plt.figure()
    plt.plot(x_true.detach().cpu().numpy(),y_true.detach().cpu().numpy(),'o',alpha=0.6,label='Date(x,y)')
    plt.plot(x_true.detach().cpu().numpy(),y_hat.detach().cpu().numpy(),'x',alpha=0.8,label='Model output on training x')
    plt.plot(x_sorted.detach().cpu().numpy(),yhat_sorted.detach().cpu().numpy(),'-',linewidth=2,label='Connected model outputs')

    plt.xlabel('x')
    plt.ylabel('value')
    plt.title('Model Outputs on Training Points')
    plt.legend()
    plt.tight_layout()
    scatter_outputs_path =os.path.join(OUT_DIR,'data_with_true_model_outputs.png')
    plt.savefig(scatter_outputs_path,dpi=150)
    plt.close()


