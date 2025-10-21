import os
import numpy as np 
from typing import List
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch.utils.data import TensorDataset,DataLoader
from model import IrisMLP
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step1:載入與探索資料

# ❌ from sklearn.datasets import load_iris 不會給你 CSV 檔案。
# ✅ 它會直接在記憶體中回傳一個 Python 物件（Bunch 類型），
# 這個物件包含 Iris 資料集的所有內容（特徵、標籤、欄位名稱等等）
# <class 'sklearn.utils._bunch.Bunch'>
# dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])

ROOT = ""
ARTIFACTS = os.path.join(ROOT,'artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)

# 資料統計摘要顯示函式
# 主要用來 快速觀察每個特徵的平均值與標準差（std）
# X 是一個 NumPy 陣列
def describe_stats(X:np.ndarray,names:List[str],title:str):
    # np.ndarray 是 NumPy 的「多維陣列（矩陣）」資料結構。

    # axis=0：每欄平均
    # m 和 s 都是 NumPy 的一維陣列
    m,s = X.mean(axis=0),X.std(axis=0)
    print(f"\n[{title}]")

    for n,mi,sd in zip(names,m,s):
        # zip() 會把它們一一對應打包成 tuple：
        # [
        #   ("sepal_length", 5.84, 0.82),
        #   ("sepal_width",  3.05, 0.43),
        #   ...
        # ]
        print(f"{n:<14s} mean={mi:8.4f} std={sd:8.4f}" )

# 載入資料
iris = load_iris()
# print(iris)

# 載入資料集：
# x 是 (150,4) 的數值矩陣
# y 是 (150,) 的標籤（0,1,2）
# 這四個特徵分別是：花萼長寬、花瓣長寬
x,y = iris.data,iris.target #x 和 y 都是 NumPy 陣列 (numpy.ndarray) 格式。
print(iris.feature_names)
feature_names = ["sepal_length","sepal_width","petal_length","petal_width"]
target_names =iris.target_names.tolist()
print(target_names)


# 用 pandas 把 x 轉成一個 DataFrame（表格格式），欄名就是特徵名稱：
df = pd.DataFrame(x,columns=feature_names)
df["target"] = y 
print("\n 前5筆資料")
print(df.head())
print("\n 類別分布:")

# 用來做「類別分布統計」
for i,name in enumerate(target_names):
    # enumerate() 是 Python 內建函式
    # 它的作用是：在迴圈中同時取得「索引」和「元素」
     print(f"{i}={name:<10s} : {(y==i).sum()} 筆")
    # y == i 表示元素逐一是否等於 i	得到一個布林arrary

describe_stats(x,feature_names,"原始資料(未標準化)")
out_csv=os.path.join(ARTIFACTS,"iris_preview.csv")

# index=False 表示 不要輸出 DataFrame 的索引欄位（只保留資料本身）
df.head(20).to_csv(out_csv,index=False)
print(f"\n->已存取20筆預覽 :{out_csv}")
print("STEP1 完成")

# Step2 切分 Train / Val / Test

# 階段	使用的 stratify	資料來源	目的
# 第一次切 (train+val / test)	stratify=y	原始完整資料	確保 test 比例與原資料一致
# 第二次切 (train / val)	stratify=y_trainval	前面切出的 trainval	確保驗證集比例與訓練資料一致

# 先切出 Test 集
X_trainval,X_test,y_trainval,y_test =train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)
# 再從 trainval 切出 Validation 集
X_train,X_val,y_train,y_val =train_test_split(
    X_trainval,y_trainval,test_size=0.2,random_state=42,stratify=y_trainval
)

# X_trainval, X_test, y_trainval, y_test
# X_train, X_val, y_train, y_val
# 👉 都是 NumPy 陣列（numpy.ndarray）格式。

print(f"切分形狀: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
print("STEP2 完成")

# Step3 標準化(只用訓練集fit) + 儲存 npz, scaler

# 第 1 行：先用訓練資料「學習平均與標準差」
# .fit(X_train) 會計算：
# 每一欄的平均值 mean_
# 每一欄的標準差 scale_
# 這一步「只用訓練集」是為了避免資料洩漏（不能偷看驗證或測試資料）
scaler = StandardScaler().fit(X_train)

# 第 2 行：把訓練資料做標準化
# 用剛剛算出的 mean_ 和 scale_ 把資料轉換成：
# (原值 - 平均) / 標準差
# 結果：每一欄的平均會變成 0、標準差變成 1
X_train_sc = scaler.transform(X_train)

# 第 3～4 行：用同一個 scaler 處理驗證與測試資料
# 這裡 不能再 fit 一次，要用 訓練集的平均與標準差 來轉換
# 這樣才確保模型在驗證/測試時使用完全相同的尺度
X_val_sc=  scaler.transform(X_val)
X_test_sc=  scaler.transform(X_test)

describe_stats(X_train, feature_names,"訓練集(標準化前)")
describe_stats(X_train_sc, feature_names,"訓練集(標準化後)")

# [存標準化資料]
# .npz 就是 把很多 NumPy 陣列一起打包壓縮存檔
# → 讓你之後可以 一次存、一包讀，很方便。
# .npz 是 NumPy 的「壓縮多陣列存檔格式」。
# 它可以同時儲存多個 .npy 陣列在一個檔案裡
npz_path= os.path.join(ARTIFACTS,"train_val_test_scaled.npz")
np.savez(
    npz_path,
    X_train_sc=X_train_sc,y_train=y_train,
    X_val_sc=X_val_sc,y_val=y_val,
    X_test_sc=X_test_sc,y_test=y_test,
    feature_names=np.array(feature_names,dtype=object),#不加 dtype=object，NumPy 會自動把所有字串轉成固定長度的「Unicode 字串型態 (<U12)」
    target_names=np.array(target_names,dtype=object),
)

# [存標準化器]
# 這兩行的目的
# 把你訓練好的 StandardScaler 物件
# 存成一個檔案（scaler.pkl），
# 以後要用時可以直接載回來，不用重新 .fit() 一次。

# 把物件存起來
# joblib.dump(scaler, scaler_path)
# 使用 joblib 的 dump 函式把 scaler（也就是你用 X_train .fit() 過的 StandardScaler）存成 .pkl 檔案
# .pkl 是「pickle」格式，用來存整個 Python 物件

scaler_path =os.path.join(ARTIFACTS,"scaler.pkl")
joblib.dump(scaler,scaler_path)


print(f"->已存標準化資料:{npz_path}")
print(f"->已存標準化器:{scaler_path}")
print("STEP3 完成")


# Step4 轉 Tensor + DataLoader 預覽

X_train_tensor = torch.tensor(X_train_sc,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)
X_val_tensor=torch.tensor(X_val_sc,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val,dtype=torch.long)

# TensorDataset(X, y)：把 X 和 y 包成一筆一筆的資料
train_data_group = TensorDataset(X_train_tensor,y_train_tensor)
val_data_group= TensorDataset(X_val_tensor,y_val_tensor)

# DataLoader(..., batch_size=16)：每次會吐出 16 筆資料（小批次）
# shuffle=True：訓練集會隨機打亂順序（避免模型記住順序）
# shuffle=False：驗證集保持原順序
# 驗證時只是「評估」模型表現，不需要也不應打亂
# 保持固定順序 → 方便對照預測與真實標籤
# 每次評估結果一致，避免隨機性影響評估
train_loader = DataLoader(train_data_group,batch_size=16,shuffle=True)
val_loader = DataLoader(train_data_group,batch_size=16,shuffle=False)


# train_loader 是你剛建立的 DataLoader，裡面有所有訓練資料（已分好 batch）
# iter(train_loader) → 建立一個「迭代器」
# next(...) → 從迭代器中取出第一組 (特徵, 標籤)
# 結果會是：
# xb = 一個 batch 的特徵資料
# shape 通常是 (batch_size, 特徵數)
# 例如 (16, 4)
# yb = 一個 batch 的標籤資料
# shape 是 (batch_size,)
# 例如 (16,)

# [重點觀念]
# shuffle=True 的 打亂時機不是在你建立 DataLoader 的當下，
# 而是在你第一次呼叫 iter(train_loader)（開始一個 epoch）時才打亂資料順序。
x_train_batch,y_train_batch=next(iter(train_loader)) 
print(f"第一個 batch:x_train_batch.shape={x_train_batch.shape}, y_train_batch.shape={y_train_batch.shape}")


# 取出 batch 裡第一筆資料的標籤
print(f"x_train_batch[0](標準化後)={x_train_batch[0].tolist()}")
print(f"y_train_batch[0](類別)={y_train_batch[0].item()}")


# 把你剛從 DataLoader 取出的那一個 batch（小批次）
# 轉成表格（pandas.DataFrame），加上標籤欄位，
# 再存成 CSV 檔，方便你用 Excel 或其他工具檢查。
batch_preview=os.path.join(ARTIFACTS,"batch_preview.csv")
pd.DataFrame(x_train_batch.numpy(),columns=feature_names).assign(label=y_train_batch.numpy()).to_csv(batch_preview,index=False)
print(f"->已存batch 預覽{batch_preview}")
print("STEP4 完成")



# Step5 定義 MLP 模型 + 輸出架構


# 計算 PyTorch 模型中「可訓練參數」的總數
# 也就是：這個模型裡 所有需要更新的權重參數 一共有幾個數值（weights / biases）。

# 你問的 -> int 是 Type Hint（型別註解） 的一種，
# 不是程式功能的一部分，只是「告訴人或工具：這個函式會回傳什麼型別」。
def count_trainable_params(model:nn.Module):

    # p.numel()
    # 回傳這個 Tensor 裡「有幾個元素」
    # 例如：
    # p.shape = (64, 4) → p.numel() = 256
    # p.shape = (64,)   → p.numel() = 64

    # 生成器（generator）這是一個 生成器（generator），而不是 list。
    
    # model.parameters() 會回傳模型中所有「可訓練參數（parameters）」的迭代器 (iterator)
    # 如果用 for p in model.parameters():
    # print(p.shape)
    # torch.Size([10, 4])  # 第一層權重 (W1)
    # torch.Size([10])     # 第一層偏置 (b1)
    # torch.Size([3, 10])  # 第二層權重 (W2)
    # torch.Size([3])      # 第二層偏置 (b2)

    # requires_grad 是什麼
    # 它的意思是：
    # 這個張量是否要在反向傳播（backpropagation）時計算梯度
    # 有些參數可能：
    # 是凍結的（不想訓練）
    # 是固定的 embedding 或預訓練權重
    
    gen = (p.numel() for p in model.parameters() if p.requires_grad)
    return sum(gen)

# [開始建立模型訓練]
device=torch.device("cpu")
# 建立模型物件
model = IrisMLP().to(device)
print(model)
print(f"可訓練參數量:{count_trainable_params(model)}")


MODELS = os.path.join(ROOT,"models")
# 新增這一行，確保資料夾存在
os.makedirs(MODELS, exist_ok=True)
arch_txt =os.path.join(MODELS,"model_arch.txt")


# [寫一個此模型功能總覽txt檔]:
# 把 模型的結構 和 可訓練參數總數
# 寫進一個文字檔（arch_txt)
with open(arch_txt,"w",encoding="utf-8")as f:
    f.write(str(model) + '\n')
    f.write(f"trainable_params={count_trainable_params(model)}\n")
print(f"-> 已存結構描述:{arch_txt}")
print("STEP5 完成") 


# STEP 6 訓練 (含早停)

# 定義「損失函式 (Loss Function)」:criterion(標準)
# 傳入 (預測分數:dtype=torch.float32,target:dtype=torch.long )
criterion = nn.CrossEntropyLoss()

# 定義「優化器 (Optimizer)」
# Adam：一種改良版的 SGD，
# 會根據歷史梯度的大小自動調整學習率 (learning rate)，
# 收斂速度通常比單純的 SGD 快。
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) #1e-3 = 1 * 10^(-3)

# [評估函式 : 去算出 1 個epoch 的 loss 及 正確率]
def evaluateFun(m, loader):
    m.eval()
    total,correct,loss_sum=0,0.0,0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device),y_batch.to(device)

            # 「logits」就是模型最後一層輸出的「未經 softmax 的原始分數 (raw scores)」
            logits = m(x_batch)
            # print('logits',logits)
            # print('y_batch',y_batch)

            # 損失計算:算出這個batch的總loss，然後加上去
            loss_sum += criterion(logits,y_batch).item() * x_batch.size(0)

            # 正確個數計算:算出這個batch有幾個正確，然後加上去
            # logits.argmax(1) → 找出這個batch中每一筆資料預測的類別索引。
            # 代表「在 dim=1 方向（每一筆樣本）找出最大分數的索引」
            # logits = tensor([
            #     [2.1, 0.5, -1.2],   # 模型預測第0類分數最高
            #     [1.3, 2.4, 0.2],    # 模型預測第1類分數最高
            #     [0.1, 0.3, 1.0],    # 模型預測第2類分數最高
            #     [2.0, 1.0, 0.1]     # 模型預測第0類分數最高
            # ])
            correct += (logits.argmax(1) == y_batch).sum().item()

            #累積樣本數:算出這個batch個數，然後加上去
            total += y_batch.size(0)
    return (loss_sum/total),(correct/total)


# 各變數的用途：
# best_state = None
# 用來存模型目前「最佳狀態」(通常是 state_dict() 的複本)。
# 剛開始還沒有最佳模型，所以是 None。

# best_val = -1.0
# 用來記錄「驗證集 (validation) 的正確率最佳表現」
# 設成 -1.0 是因為我們希望後面第一次驗證結果一定會比它好（假設準確率 ≥ 0）。

# patience = 15
# 代表「容忍連續多少次表現沒有進步」
# 如果超過 15 次 epoch 都沒有改善，就觸發 early stopping，停止訓練。

# bad = 0
# 記錄「已經連續幾次沒有進步」
# 每當 val_acc 沒有變好，就 bad += 1；有變好就 bad = 0。

best_state,best_val,patience,bad = None, -1.0, 15, 0 
hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}

# [訓練開始]
for ep in range(1, 201): 
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device),y_batch.to(device)
        logits=model(x_batch)
        

        # 這是這個 batch 的「平均損失 (平均 cross-entropy)」是一個數字
        loss_batch_avg =  criterion(logits, y_batch)

        #⬅️ 反向傳播 + 參數更新
        optimizer.zero_grad()
        loss_batch_avg.backward()
        optimizer.step()

        # 更新全域變數 total, correct, loss_sum

        # loss_batch_sum一個batch總loss
        loss_batch_sum =  loss_batch_avg.item() * x_batch.size(0) 
        loss_sum += loss_batch_sum

        # 正確個數計算:算出這個batch有幾個正確，然後加上去
        # logits.argmax(1) → 找出這個batch中每一筆資料預測的類別索引。
        correct += (logits.argmax(1) == y_batch).sum().item()

        total += y_batch.size(0)

    #[做完一個epoch用驗證集去驗證一下]
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

        # 建立一個新的字典，
        # 內容是模型所有參數的字典 (state_dict) 的 名稱與值，
        # 但每個參數都被「搬回 CPU」上。
        # model.state_dict()
        # 會回傳一個「字典 (dict)」，
        # 裡面包含 模型所有可學習的參數與緩衝區：
        # key (字串)	value (Tensor)
        # 'layer1.weight'	Tensor([...])
        # 'layer1.bias'	Tensor([...])
        # 'layer2.weight'	Tensor([...])
        # 'layer2.bias'	Tensor([...])
        best_state ={key:value.cpu() for key,value in model.state_dict().items()}

        bad=0
    else:
        bad += 1
        if bad >= patience:
            print(f"早停：{patience} 個epochs 未提升")
            break

# 載回最佳
if best_state is not None:
    model.load_state_dict(best_state)


# Step7 畫圖及儲存最佳模型權重:

# === 畫訓練/驗證曲線 ===
x_range = np.arange(1,len(hist["tr_loss"]) + 1)

plt.figure(figsize=(8, 4))

# loss 曲線
plt.plot(x_range,hist["tr_loss"],label="train_loss")
plt.plot(x_range,hist["va_loss"],label="va_loss")

# acc 曲線
plt.plot(x_range,hist["tr_acc"],label="tr_acc")
plt.plot(x_range,hist["va_acc"],label="va_acc")

plt.xlabel("epoch")
plt.ylabel("value")
plt.title("Training Curves")
plt.legend() #顯示圖例 (Legend)
plt.tight_layout()

PLOTS = os.path.join(ROOT, "plots")

# 確保資料夾存在
os.makedirs(PLOTS, exist_ok=True)

# 儲存圖表
curve_path = os.path.join(PLOTS, "curves.png")
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"✅已存訓練/驗證曲線: {curve_path}")

# === 儲存最佳模型權重 ===
best_path = os.path.join(MODELS, "best.pt")
torch.save(model.state_dict(),best_path)
print(f"✅ 已存最佳權重: {best_path}")
print("STEP 7 ✅ 完成")

# STEP 8 用測試集模擬 及 混淆矩陣矩陣分析圖

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
# interpolation="nearest" 每個格子一個純色方塊（最清晰，常用於混淆矩陣
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

# 設定軸刻度 (Axis Ticks)
ticks = np.arange(len(target_names)) #numpy array([0, 1, 2])

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