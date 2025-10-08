# -*- coding: utf-8 -*-
"""
plot_metrics_fixed.py
讀取固定路徑的 loss.csv，畫出 train/test 的 loss 與 acc 折線圖
使用：
    python plot_metrics_fixed.py
"""
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

#[Step 1] 檔案設定
# 目的：定義路徑與環境，讓整個程式知道要從哪裡讀檔、輸出到哪裡。

   # === 檔案路徑設定(這裡參數寫死) === 
CSV_PATH = Path("result/loss.csv")
OUT_PATH = Path("result/metrics.png")

#[Step 2] 讀取資料
# 目的：安全讀取訓練結果 CSV，並驗證欄位格式是否正確。

   # === 讀取 CSV 檔 ===
def read_metrics(csv_path:Path):
    # Python 的一種**快速建立字典（dict）**的語法
    cols={k:[] for k in ["epoch","train_loss","val_loss","train_acc","val_acc"]}

    # 嘗試 UTF-8-SIG 與 UTF-8 編碼
    try:
        f=open(csv_path,"r",encodinh="utf-8-sig")
        first_try = f
    except:
        f=open(csv_path,"r",encodinh="utf-8")
        first_try = f
    
    with first_try as ff:
        # 建立一個可以逐行讀取 CSV 並自動用欄位名稱當 key 的讀取器 (reader)
        reader = csv.DictReader(ff)
    
        if not reader.fieldnames:
            # 程式中主動拋出錯誤（raise an exception）
            raise RuntimeError("CSV沒有表頭，請確認內容")
        
        # fmap 的目的：建立欄位「翻譯表」，欄位名稱轉小寫去空白
        fmap ={k.strip().lower(): k for k in  reader.fieldnames}
        need=["epoch","train_loss","val_loss","train_acc","val_acc"]
        miss=[k for k in need if k not in fmap ]

        if miss:
            raise RuntimeError(f"缺少必要欄位:{miss};找到的欄位有:{reader.fieldnames}")
        
        for row in reader :      
            try:
                e=float(row[fmap["epoch"]])
                tl= float(row[fmap["train_loss"]]) 
                vl=float(row[fmap["val_loss"]]) 
                ta=float(row[fmap["train_acc"]]) 
                va=float(row[fmap["val_acc"]]) 

            except Exception:
                continue #忽略錯誤行

            cols["epoch"].append(e)
            cols["train_loss"].append(tl)
            cols["val_loss"].append(vl)
            cols["train_acc"].append(ta)
            cols["val_acc"].append(va)
            # ex:大概長這樣
            #{
            #   "epoch": [1.0, 2.0, 3.0, ...],
            #   "train_loss": [0.5, 0.4, 0.35, ...],
            #   "val_loss": [0.6, 0.55, 0.5, ...],
            #   "train_acc": [0.8, 0.85, 0.88, ...],
            #   "val_acc": [0.78, 0.82, 0.84, ...]
            # }
        if not cols["epoch"]:
            raise RecursionError("沒有成功讀到任何資料，請檢查 CSV 內容格式")
        
        return cols #回傳整部字典

#[Step 3] 輔助函式
# 目的：找出最小 / 最大指標值的索引，用於標註圖上最佳點。
def argmin(lst):
    bi ,bv =None,math.inf
    for i,v in enumerate(lst):
        if v < bv:
            bi,bv =i,v
    return bi

def argmax(lst):
    bi ,bv =None,math.inf
    for i,v in enumerate(lst):
        if v < bv:
            bi,bv =i,v
    return bi


#[Step 4] 主繪圖流程
def main():
    # 「如果 result/ 資料夾不存在，就幫我自動建立它。如果它已經存在，就略過，不報錯。」
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    data= read_metrics(CSV_PATH)
    epochs=data["epoch"]

    index_min_val_loss=argmin(data["val_loss"])
    index_max_val_acc=argmin(data["val_acc"])
    ep_min_val_loss=epochs[index_min_val_loss]
    ep_max_val_loss=epochs[index_max_val_acc]

    fig = plt.figure(figsize=(12,5),constrained_layout=True)#constrained_layout=True:讓 Matplotlib 自動調整子圖之間的間距，避免文字、標題或標籤重疊。

    # ----loss 圖------
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs,data["train_loss"],label="train_loss",lineWidth=2)
    ax1.plot(epochs,data["val_loss"],label="val_loss",lineWideh=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (train vs valicate)")
    ax1.grid(True,linestyle="--",alpha=0.4)
    ax1.legend(loc="best") #在圖上自動選一個不會擋住線條的位置，顯示圖例說明。
      #標記最小 loss
    ax1.axvline(ep_min_val_loss,linestyle='--',alpha=0.6) #全名是 “Axes Vertical Line”(垂直線) 
    # annoteate:在圖上指定一個點 (x, y)，並在旁邊顯示說明文字、加箭頭指向該點。
    ax1.annotate(
        f"min val_loss @ ep {ep_min_val_loss}\n(val_loss={data['val_loss'][index_min_val_loss]})",
        xy=(ep_min_val_loss,data['val_loss'][index_min_val_loss]),
        xytext=(0,15),#相對於該點的位移
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->",alpha=0.6),
        fontsize=9,
        ha="center",
        annotation_clip=False #即使箭頭在圖邊界外，也強制顯示註解，不被裁掉。
    )
    ax1.margins(x=0.05,y=0.10) #調整圖表邊界留白（margin）的設定指令
#[Step 5] 主執行點
if __name__ == "__main__":
    main()


