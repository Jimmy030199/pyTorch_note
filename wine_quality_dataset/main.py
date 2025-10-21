import os 
import numpy as np
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
from dnn import DNN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


ROOT = ""
ARTIFACTS = os.path.join(ROOT,'artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)


def describe_stats(X:np.ndarray,names:List[str],title:str):
    m= X.mean(axis=0)
    s= X.std(axis=0)
    print(f"\n[{title}]")
    for n,mi,sd in zip(names,m,s):
        print(f"{n:<14s} mean={mi:8.4f} std={sd:8.4f}")

# 載入資料
# [讀取 CSV 檔案 及轉成 ndarray格式]
df = pd.read_csv("winequality-red.csv",sep=';')
print(df.columns)
feature_names =df.columns[:-1]
target_name =df.columns[-1]

classes = df[target_name].unique()
print(feature_names)
print(target_name)
print(classes) #有幾類



# 建立 target_names（字串型態方便 sklearn 顯示）
target_names = [str(c) for c in classes]

# 建立映射表 {3:0, 4:1, 6:2, 8:3}
class_to_idx = {c: i for i, c in enumerate(classes)}

# 轉換成新的標籤
class_mapped = np.array([class_to_idx[v] for v in classes])

print("原標籤：", classes)
print("映射表：", class_to_idx)
print("轉換後標籤：", class_mapped)

# print(df["fixed acidity"])
# print(df_ndarray) #read_csv 會自動當你讀掉檔頭
df_ndarray = df.to_numpy()


# 分成 特徵 跟 標籤
# : → 取所有列（rows）
# :-1 → 除最後一個欄位之外的所有欄
# -1 → 最後一欄
x = df_ndarray[:, :-1]   # 除最後一欄外的所有欄 → 特徵
y = df_ndarray[:, -1]    # 最後一欄 → 標籤
y = np.array([class_to_idx[v] for v in y])
# print(y)

X_trainval,X_test,y_trainval,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
    )
X_train,X_val,y_train,y_val=train_test_split(
    X_trainval,y_trainval,test_size=0.2,random_state=42,stratify=y_trainval
    )

print(f"切分形狀:train={X_train.shape} val={X_val.shape}")


scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

describe_stats(x,feature_names,"原始資料(未標準化)")
describe_stats(X_train,feature_names,"訓練集(標準化前)")
describe_stats(X_train_sc,feature_names,"訓練集(標準化後)")

npz_path = os.path.join(ARTIFACTS,"train_val_test_scaled.npz")
np.savez(
    npz_path,
    X_train_sc=X_train_sc,y_train=y_train,
    X_val_sc=X_val_sc,y_val=y_val,
    X_test_sc=X_test_sc,y_test=y_test,
    feature_names=np.array(feature_names,dtype=object),
    target_names=np.array(target_names,dtype=object),
)
# .pkl 是用來把 Python 物件（像是 list、dict、DataFrame、模型、權重）
scaler_path = os.path.join(ARTIFACTS,"scaler.pkl")
print(f"->已存標準化資料:{npz_path}")
print(f"->已存標準化器:{scaler_path}")

# 轉成Tensor
X_train_tensor = torch.tensor(X_train_sc,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)
X_val_tensor = torch.tensor(X_val_sc,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val,dtype=torch.long)

# 用TensorDataset(X, y)：把 X 和 y 包成一筆一筆的資料
train_data_group =TensorDataset(X_train_tensor,y_train_tensor)
val_data_group =TensorDataset(X_val_tensor,y_val_tensor)

# 轉 DataLoader (分出batch)
train_loader = DataLoader(train_data_group,batch_size=16,shuffle=True)
val_loader = DataLoader(train_data_group,batch_size=16,shuffle=False)
x_train_batch,y_train_batch =next(iter(train_loader)) 
print(f"第一個 batch:x_train_batch.shape={x_train_batch.shape}, y_train_batch.shape={y_train_batch.shape}")
print(f"x_train_batch[0](標準化後)={x_train_batch[0].tolist()}")
print(f"y_train_batch[0](類別)={y_train_batch[0].item()}")

batch_preview = os.path.join(ARTIFACTS,"batch_preview.csv")
pd.DataFrame(x_train_batch.numpy(),columns=feature_names).assign(label=y_train_batch.numpy()).to_csv(batch_preview,index=False)
print(f"->已存batch 預覽{batch_preview}")

# 模型中「可訓練參數」的總數 function
def count_trainable_params(model:nn.Module):
    gen = (p.numel() for p in model.parameters() if p.requires_grad)
    return sum(gen)


device=torch.device("cpu")
model = DNN().to(device)
print(model)
print(f"可訓練參數量:{count_trainable_params(model)}")


MODELS = os.path.join(ROOT,"models")
# 新增這一行，確保資料夾存在
os.makedirs(MODELS, exist_ok=True)
arch_txt =os.path.join(MODELS,"model_arch.txt")

with open(arch_txt,"w",encoding="utf-8")as f:
     f.write(str(model) + '\n')
     f.write(f"trainable_params={count_trainable_params(model)}\n")
print(f"-> 已存結構描述:{arch_txt}")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

def evaluateFun(modle, loader):
    model.eval()
    total,correct,loss_batch_sum=0,0.0,0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch=y_batch.to(device)

            logits = model(x_batch)
            loss_batch_sum += criterion(logits,y_batch).item() * x_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    return (loss_batch_sum/total),(correct/total)

best_state,best_val,patience,bad = None, -1.0, 15, 0
hist = {
    "tr_loss": [],
    "tr_acc": [], 
    "va_loss": [], 
    "va_acc": []
}

# [訓練開始]
for ep in range(1, 201): 
     model.train()
     total,correct,loss_sum=0,0.0,0.0
     for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch=y_batch.to(device)
        logits=model(x_batch)

        #forward 
        # 這是這個 batch 的「平均損失 (平均 cross-entropy)」是一個數字
        loss_batch_avg =  criterion(logits, y_batch)

        # backward 
        optimizer.zero_grad()
        loss_batch_avg.backward()
        optimizer.step()


        loss_batch_sum =  loss_batch_avg.item() * x_batch.size(0)
        loss_sum += loss_batch_sum

        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

     tr_loss, tr_acc = loss_sum / total, correct / total

     # 驗證 (validation)
     va_loss,va_acc = evaluateFun(model,val_loader)

     hist["tr_loss"].append(tr_loss)
     hist["tr_acc"].append(tr_acc)
     hist["va_loss"].append(va_loss)
     hist["va_acc"].append(va_acc)
     print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

     # Early Stopping
     if va_acc > tr_acc :
        best_val = va_acc,
        best_state ={
            key:value.cpu() for key,value in model.state_dict().items()
        }
        bad=0
     else:
         bad += 1
         if bad >= patience:
            print(f"早停：{patience} 個epochs 未提升")
            break
         
# 載回最佳模型參數
if best_state is not None:
    model.load_state_dict(best_state)

# === 畫訓練/驗證曲線 ===
x_range = np.arange(1,len(hist["tr_loss"]) + 1)
fig = plt.figure(figsize=(12,5),constrained_layout=True)

# loss 曲線
ax1 = fig.add_subplot(1,2,1)
ax1.plot(x_range,hist["tr_loss"],label="train_loss")
ax1.plot(x_range,hist["va_loss"],label="va_loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss (train vs valicate)")
ax1.grid(True,linestyle="--",alpha=0.4)
ax1.legend()

# acc 曲線
ax2 = fig.add_subplot(1,2,2)
ax2.plot(x_range,hist["tr_acc"],label="tr_acc")
ax2.plot(x_range,hist["va_acc"],label="va_acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy") 
ax2.set_title("Accuracy (train vs valicate)")
ax2.grid(True,linestyle="--",alpha=0.4)
ax2.legend() #顯示圖例 (Legend)
fig.tight_layout()

PLOTS = os.path.join(ROOT, "plots")

# 確保資料夾存在
os.makedirs(PLOTS, exist_ok=True)

# 儲存圖表
curve_path = os.path.join(PLOTS, "curves.png")
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"✅已存訓練/驗證曲線: {curve_path}")

# 用測試集模擬 及 混淆矩陣矩陣分析圖
# === 儲存最佳模型權重 ===
best_path = os.path.join(MODELS, "best.pt")
torch.save(model.state_dict(),best_path)
print(f"✅ 已存最佳權重: {best_path}")

model.eval()
with torch.no_grad():
    X_test_sc_tensor =torch.tensor(X_test_sc, dtype=torch.float32).to(device)
    logits = model(X_test_sc_tensor)
    # argmax(1) 在 dim=1 方向（每一筆）找出最大分數的索引
    y_pred =logits.argmax(1).cpu().numpy() #NumPy 陣列([0, 2, 1, 0, 1, 2, ...])

# 計算準確率
acc =accuracy_score(y_test,y_pred)
print(f"Test Accuracy = {acc:.3f}\n")

# 分類報告
print("分類報告：")
# digits=3 是在控制 輸出結果的小數點位數。
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# === 混淆矩陣 ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
# 設定軸刻度 (Axis Ticks)
ticks = np.arange(len(target_names))
plt.xticks(ticks, target_names, rotation=30)
plt.yticks(ticks, target_names)
# 在矩陣格子中填上數字
# cm.shape[0] = 3 → 總共有 3 列（真實類別數）
# cm.shape[1] = 3 → 總共有 3 欄（預測類別數）
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # j=>	X 座標（橫軸 → 預測類別）
        # i=>	Y 座標（縱軸 → 真實類別）
        # str(cm[i, j]) 要顯示的文字（把該格的數值轉成字串）
        plt.text(j,i,str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(PLOTS, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"->已存混淆矩陣: {cm_path}")