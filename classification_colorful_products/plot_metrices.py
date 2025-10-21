from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

CSV_PATH =Path("result/loss.csv")
OUT_PATH = Path("result/metrics.png")

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

def argmin(lst):
    bi,bv = None,math.inf
    for i,v in enumerate(lst):
        if v <bv:
            bi,bv = i,v
    return bi

def argmax(lst):
    bi,bv = None,-math.inf
    for i,v in enumerate(lst):
        if v >bv:
            bi,bv = i,v
    return bi

def main():
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    data=read_metrics(CSV_PATH)
    epochs=data["epoch"]

    index_min_val_loss = argmin(data["val_loss"])
    index_max_val_acc = argmax(data["val_acc"])

    ep_min_val_loss = epochs[index_min_val_loss]
    ep_max_val_acc = epochs[index_max_val_acc]

    fig=plt.figure(figsize=(12,5),constrained_layout=True)

    # ---loss 圖
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs,data["train_loss"],label="train_loss",linewidth=2) 
    ax1.plot(epochs,data["val_loss"],label="val_loss",linewidth=2) 
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("LOSS (train vs valicate)")
    ax1.grid(True,linestyle="--",alpha=0.4)
    ax1.legend(loc="best")
    ax1.axvline(ep_min_val_loss,linestyle="--",alpha=0.6)
    ax1.annotate(
        f"min val_loss @ ep {ep_min_val_loss}\n (val_loss={data['val_loss'][index_min_val_loss]:.4f})",
        xy=(ep_min_val_loss,data["val_loss"][index_min_val_loss]),
        xytext=(0,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle='->',alpha=0.6),
        fontsize=9,
        ha="center",
        annotation_clip=False
    )
    ax1.margins(x=0.05,y=0.10)

    # accuracy 圖
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs,data["train_acc"],label="train_acc",linewidth=2) 
    ax2.plot(epochs,data["val_acc"],label="val_acc",linewidth=2) 
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy (train vs valicate)")
    ax2.grid(True,linestyle="--",alpha=0.4)
    ax2.legend(loc="best")
    ax2.axvline(ep_max_val_acc,linestyle="--",alpha=0.6)
    ax2.annotate(
        f"max val_acc @ ep {ep_max_val_acc}\n (val_acc={data['val_acc'][index_max_val_acc]:.4f})",
        xy=(ep_max_val_acc,data["val_acc"][index_max_val_acc]),
        xytext=(0,15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle='->',alpha=0.6),
        fontsize=9,
        ha="center",
        annotation_clip=False
    )
    ax2.margins(x=0.05,y=0.10)

    fig.suptitle("Training Metrics",fontsize=14,y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PATH,dpi=150)
    print(f"[ok]圖片已輸出:{OUT_PATH}")

if __name__ == "__main__":
    main()


