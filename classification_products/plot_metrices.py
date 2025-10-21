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
    # (1) 🔹建立空的結果字典
    # Python 的一種**快速建立字典（dict）**的語法
    cols={k:[] for k in ["epoch","train_loss","val_loss","train_acc","val_acc"]}

    # (2) 🔹嘗試不同編碼打開 CSV
    # 嘗試 UTF-8-SIG 與 UTF-8 編碼
    try:
        f=open(csv_path,"r",encoding="utf-8-sig")
        first_try = f
    except:
        f=open(csv_path,"r",encoding="utf-8")
        first_try = f
    
    with first_try as ff:
        # (3) 🔹建立 CSV 讀取器
        # 建立一個可以逐行讀取 CSV 並自動用欄位名稱當 key 的讀取器 (reader)
        # 這行建立了一個 CSV 讀取器 (DictReader)。
        # 它會把 CSV 檔的每一行轉成 字典 (dict) 的形式。
        # epoch,train_loss,val_loss,train_acc,val_acc
        # 1,0.523,0.612,0.812,0.798
        # 2,0.421,0.587,0.856,0.827
            # 讀成
        # {'epoch': '1', 'train_loss': '0.523', 'val_loss': '0.612', 'train_acc': '0.812', 'val_acc': '0.798'}
        # {'epoch': '2', 'train_loss': '0.421', 'val_loss': '0.587', 'train_acc': '0.856', 'val_acc': '0.827'}

        reader = csv.DictReader(ff)
    
        # (4) 🔹檢查 CSV 表頭是否存在
        if not reader.fieldnames:
            # 程式中主動拋出錯誤（raise an exception）
            raise RuntimeError("CSV沒有表頭，請確認內容")
        
        # (5) 🔹建立欄位翻譯表 fmap
        # fmap 的目的：建立欄位「翻譯表」，欄位名稱轉小寫去空白
        # fmap = {
        #     'epoch': ' Epoch ',
        #     'train_loss': ' Train_Loss ',
        #     'val_loss': ' Val_Loss ',
        #     'train_acc': ' Train_Acc ',
        #     'val_acc': ' Val_Acc '
        # }
        fmap ={k.strip().lower(): k for k in  reader.fieldnames}

        # (6) 🔹檢查是否缺少必要欄位
        need=["epoch","train_loss","val_loss","train_acc","val_acc"]
        miss=[k for k in need if k not in fmap ]

        if miss:
            raise RuntimeError(f"缺少必要欄位:{miss};找到的欄位有:{reader.fieldnames}")
        
        # 在你的程式裡，這個 for row in reader: 是要：
        # 依序拿到每一筆資料
        # 把裡面的數值轉成浮點數（float）
        # 儲存到 cols 字典裡面
        for row in reader :      
            try:
                # 拿出一行的資料
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
    bi, bv = None, -math.inf   # 🚩 起始值應該是 -∞
    for i, v in enumerate(lst):
        if v > bv:              # 🚩 改成「大於」
            bi, bv = i, v
    return bi


#[Step 4] 主繪圖流程
def main():
    # 「如果 result/ 資料夾不存在，就幫我自動建立它。如果它已經存在，就略過，不報錯。」
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    data= read_metrics(CSV_PATH)
    epochs=data["epoch"]

    index_min_val_loss=argmin(data["val_loss"])
    index_max_val_acc=argmax(data["val_acc"])
    ep_min_val_loss=epochs[index_min_val_loss]
    ep_max_val_acc=epochs[index_max_val_acc]

    fig = plt.figure(figsize=(12,5),constrained_layout=True)#constrained_layout=True:讓 Matplotlib 自動調整子圖之間的間距，避免文字、標題或標籤重疊。

    # ----loss 圖------
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs,data["train_loss"],label="train_loss",linewidth=2)
    ax1.plot(epochs,data["val_loss"],label="val_loss",linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (train vs valicate)")
    ax1.grid(True,linestyle="--",alpha=0.4)
    ax1.legend(loc="best") #在圖上自動選一個不會擋住線條的位置，顯示圖例說明。
      #標記最小 loss
    ax1.axvline(ep_min_val_loss,linestyle='--',alpha=0.6) #全名是 “Axes Vertical Line”(垂直線) 
    # annoteate:在圖上指定一個點 (x, y)，並在旁邊顯示說明文字、加箭頭指向該點。
    ax1.annotate(
        f"min val_loss @ ep {ep_min_val_loss}\n(val_loss={data['val_loss'][index_min_val_loss]:.4f})",
        xy=(ep_min_val_loss,data['val_loss'][index_min_val_loss]),
        xytext=(0,15),#相對於該點的位移
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->",alpha=0.6),
        fontsize=9,
        ha="center",
        annotation_clip=False #即使箭頭在圖邊界外，也強制顯示註解，不被裁掉。
    )
    ax1.margins(x=0.05,y=0.10) #調整圖表邊界留白（margin）的設定指令

    # --- Accuracy 圖 ---
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs,data["train_acc"],label="train_acc",linewidth=2)
    ax2.plot(epochs,data["val_acc"],label="val_acc",linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy") 
    ax2.set_title("Accuracy (train vs valicate)")
    ax2.grid(True,linestyle="--",alpha=0.4)
    ax2.legend(loc="best")

    # 標註最高 accuracy
    ax2.axvline(ep_max_val_acc, linestyle="--", alpha=0.6)
    ax2.annotate(
        f"max test_acc @ ep {ep_max_val_acc}\n(val_acc={data['val_acc'][index_max_val_acc]:.4f})",
        xy=(ep_max_val_acc, data["val_acc"][index_max_val_acc]),
        xytext=(0, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9, ha="center", annotation_clip=False,
    )
    ax2.margins(x=0.05, y=0.10) 


    fig.suptitle("Training Metrics",fontsize=14,y=0.98)  #y=0.98 是為了把標題往上微調一點
    fig.tight_layout() #自動調整子圖（subplot）之間的間距與邊界，避免文字、標籤、圖例重疊
    fig.savefig(OUT_PATH,dpi=150)
    print(f"[ok]圖片以輸出:{OUT_PATH}")

#[Step 5] 主執行點
if __name__ == "__main__":
    main()


