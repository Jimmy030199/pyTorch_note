import numpy as np 
import pandas as pd
import torch
from torch import nn
from typing import List
import csv
from pathlib import Path
import math

# [若你想同時分出 X, y（常見於 sklearn）]
# 說明:
# 以利後續DataLoader前，
# 會用TensorDataset(X, y)：把 X 和 y 包成一筆一筆的資料
# 用 兩個 Tensor（X_train_tensor 和 y_train_tensor）
# 建立一個新的 Dataset，
# 每一筆資料都會對應成 (特徵, 標籤) 這樣的「兩欄」結構。
# 間接告知DataLoader 誰是標籤欄
df = pd.read_csv("?????.csv",sep=';')
target_name = "emotion" #表示標籤的欄位名
X = df.drop(columns=[target_name]).to_numpy()
y = df[target_name].to_numpy()

# 建立映射表 {3:0, 4:1, 6:2, 8:3}
target_name = "emotion"
classes = df[target_name].unique()
class_to_idx = {c: i for i, c in enumerate(classes)}
y = df[target_name].to_numpy()
y = np.array([class_to_idx[v] for v in y])

# 主要用來 快速觀察每個欄位特徵的平均值與標準差（std）
def describe_stats(X:np.ndarray,names:List[str],title:str):
    m,s = X.mean(axis=0),X.std(axis=0)
    print(f"\n[{title}]")

    for n,mi,sd in zip(names,m,s):
        print(f"{n:<14s} mean={mi:8.4f} std={sd:8.4f}" )

# 模型中「可訓練參數」的總數 function
def count_trainable_params(model:nn.Module):
    gen = (p.numel() for p in model.parameters() if p.requires_grad)
    return sum(gen)


# 計算模型1個epoch整體在驗證/測試資料上的平均 loss 和 accuracy
def evaluateFun(modle, loader):
    model.eval()
    total,correct,loss_batch_sum=0,0.0,0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch=y_batch.to(device)

            # logits 是一整個batch 的預測分數集
            # [
            # [0.5,0.6,0.3]
            # [0.3,0.5,0.7]
            # [0.2,0.3,0.1]
            # ]
            logits = model(x_batch)
            loss_batch_sum += criterion(logits,y_batch).item() * x_batch.size(0)
            # argmax 會回傳「最大值所在的位置索引」
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    return (loss_batch_sum/total),(correct/total)



# 這個函式是「把訓練結果 CSV 檔（epoch / loss / acc）」
# 讀回來成 Python 可用的數據格式（字典），
# 讓你能快速畫圖或分析模型收斂狀況。
def read_metrics(csv_path:Path):
    cols={k:[] for k in ["epoch","train_loss","val_loss","train_acc","val_acc"]}
    try:
        f=open(csv_path,"r",encoding="utf-8-sig")
        first_try = f
    except:
        f=open(csv_path,"r",encoding="utf-8")
        first_try = f
    
    with first_try as ff :
        reader = csv.DictReader(ff)
        if not reader.fieldnames:
            raise RuntimeError("csv 沒有表頭")
    
        for row in reader :
            # row = {'epoch':'1', 
            # 'train_loss':'1.2859', 
            # 'val_loss':'1.0593', 
            # 'train_acc':'0.5121', 
            # 'val_acc':'0.6184'}
            try:
                e=float(row["epoch"])
                tl=float(row["train_loss"])
                vl=float(row["val_loss"])
                ta=float(row["train_acc"])
                va=float(row["val_acc"])
            except Exception:
                continue

            cols["epoch"].append(e)
            cols["train_loss"].append(tl)
            cols["val_loss"].append(vl)
            cols["train_acc"].append(ta)
            cols["val_acc"].append(va)

        if not cols["epoch"]:
            raise RecursionError("沒有成功獨到資料") 
        return cols
    
# 回傳一個 list（或可迭代物件）中最小值的索引 (index)
def argmin(lst):
    bi,bv = None,math.inf
    for i,v in enumerate(lst):
        if v <bv:
            bi,bv = i,v
    return bi

# 找出 list 或序列中最大值的索引位置 (index)。
def argmax(lst):
    bi,bv = None,-math.inf
    for i,v in enumerate(lst):
        if v >bv:
            bi,bv = i,v
    return bi