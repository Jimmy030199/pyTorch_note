import os
import numpy as np 
from typing import List
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Step1:載入與探索資料

ROOT = "dnn_iris"
ARTIFACTS = os.path.join(ROOT,'artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)

# 資料統計摘要顯示函式
# 主要用來 快速觀察每個特徵的平均值與標準差（std）
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