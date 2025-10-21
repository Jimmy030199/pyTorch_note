# -*- coding: utf-8 -*-
"""
test.py
載入訓練好的模型，隨機挑選幾張 Fashion-MNIST 圖片做預測並繪出結果
"""
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torchvision import datasets,transforms
from cnn import CNN
import torch.nn.functional as F
import os

# (1) 匯入模組與設定參數
# ========== 可調參數 ==========
DATA_DIR="data"
OUT_DIR="result"
CKPT_PATH=Path(OUT_DIR) / "best_cnn.pth" # 模型檔路徑 (checkpoint)
ROWS,COLS= 2,5 # 輸出圖片網格 (共 ROWS×COLS 張)
SEED=42
# 這兩個值（0.2861 與 0.3530）
# 是由 整個 Fashion-MNIST 訓練集 計算得來的統計量：
# 這代表：
# 整體平均灰階亮度 = 0.2861
# 整體亮度變異程度 = 0.3530
MEAN,STD= 0.2861, 0.3530 # Fashion-MNIST 正規化參數
FIGSIZE=(16,8)
TITLE_FONTSIZE=11

# Windows 下常見中文字體（用來顯示中文標題）
ZH_FONT_CANDIDATES =[
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]

# 類別名稱
CLASS_NAMES=[
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# ==============================================

# (2) 定義輔助函式
# │       ├─ set_seed()     → 固定隨機性
# │       ├─ set_zh_font()  → 中文字體設定
# │       └─ denorm_img()   → 去正規化圖片
# ========== 輔助函式 ==========

"""固定隨機種子，確保結果可重現"""
def set_seed(seed):
    random.seed(seed) #固定 Python 內建的亂數模組 random 的種子。
    torch.manual_seed(seed) #固定 CPU 上 PyTorch 的亂數種子。
    torch.cuda.manual_seed_all(seed) #固定 GPU 上 的亂數種子。

"""設定中文字體 (避免 matplotlib 顯示亂碼)"""
def set_zh_font():
    for path in ZH_FONT_CANDIDATES:
        if Path(path).exists():
            # plt.rcParams 是 Matplotlib 的全域設定字典
            plt.rcParams["font.sans-serif"] =[font_manager.FontProperties(fname=path).get_name()]
            plt.rcParams["axes.unicode_minus"]=False #防止 負號（−） 被顯示成亂碼（例如方塊 □）
            print(f"[INFO]已啟用中文字體:{path}")
            return
    print("[WARN] 找不到中文字體，中文可能無法正常顯示。")

"""把 Normalize 後的圖片轉回 0~1 區間方便顯示"""
# BatchNorm2d 處理的是 模型的特徵圖 (feature maps)
# denorm_img() 處理的是 原始影像輸入 (input images)
# denorm_img() 跟這沒有關係，它是用來把輸入圖片轉回可顯示的亮度，
# 不是處理模型輸出。
def denorm_img(x):
    # x 是一張經過 Normalize（標準化）後的灰階影像張量
    # 目的:→ 把 -2~2 區間的像素值轉回 0~1。
    x = x * STD + MEAN

    # 把 x 的每個元素限制在 0 和 1 之間。
    # 這裡的 x 是一張「反正規化後的影像張量」。
    # 反正規化後的像素值理論上應該介於 0～1 之間，
    # 但由於浮點數誤差或計算順序，
    # 可能會出現極少數像素略小於 0 或略大於 1（例如 -0.003 或 1.002）。
    # → 為了避免 imshow() 顯示時出現顏色錯誤，
    # 就用 .clamp(0,1) 把它強制截斷回合法亮度範圍。
    return x.clamp(0,1)

# ====================================================

# (3) main() 主流程
# │       ├─ 初始化 (隨機種子、字體、裝置)
# │       ├─ 載入測試資料集
# │       ├─ 載入訓練後模型
# │       ├─ 隨機選取測試圖片
# │       ├─ 預測分類與信心值
# │       ├─ 建立圖片網格顯示預測結果
# │       ├─ 輸出結果圖 test_grid.png
def main():
    set_seed(SEED)
    set_zh_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 載入測試資料集 ---
    # 1️⃣ ToTensor()：把影像轉成 [0,1] 的 tensor
    # 2️⃣ Normalize()：再把數值轉成以 mean 為中心、std 為尺度的分布
    # 前處理
    tfm = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((MEAN,),(STD,))])
    # test_set 是一個 Dataset 物件 不是DataLoader 所以沒有batch這一維度
    test_set = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=tfm
        )
    
    # --- 載入模型 ---
    model = CNN(num_classes =10).to(device)

    if CKPT_PATH.exists():
        #torch.load(...)這個函式會把 .pth 檔重新載回記憶體（反序列化）。
        # 載回來後是一個 Python 字典 (dict)
        ckpt = torch.load(CKPT_PATH,map_location=device) 
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] 已載入模型權重: {CKPT_PATH}")
    else:
        print(f"[WARN] 模型檔案不存在: {CKPT_PATH}，使用隨機初始化權重！")

    model.eval()

    # --- 隨機挑選 N 張圖片 ---
    N = COLS * ROWS
    indices = list(range(len(test_set))) # 建立索引
    random.shuffle(indices) # 打亂順序 (set_seed(SEED) 已固定seed 所以每次一樣)
    indices = indices[:N] # 取前 N 張

    images = []  # 儲存取出來的原始圖片（tensor 格式）ex:[tensor(1,28,28), tensor(1,28,28), ...]
    gts = []     # 儲存真實標籤（Ground Truth）ex:[9, 2, 5, 1, ...]
    preds = []   #儲存模型預測的類別編號 ex:[9, 2, 3, 1, ...]
    confs = []   #儲存每次預測的信心值（softmax 機率）ex:[0.987, 0.954, 0.743, ...]

    with torch.no_grad():
        for idx in indices:
            img_t,label = test_set[idx]
            # 1️⃣ unsqueeze(0) → 增加 batch 維度 [1, 1, 28, 28]
            # 2️⃣ .to(device) → 放到 GPU/CPU 執行
            # logits 的 shape 是 [batch_size, num_classes] → [1, 10]
            logits=model(img_t.unsqueeze(0).to(device))

            # 轉成「每一類的機率分布」，讓我們可以看出模型信心
            # dim=0在 batch 維度上做
            # dim=1 在類別維度上做
            #[0]因為 長[[0.7054, 0.2369, 0.0577.......]]
            prob = F.softmax(logits,dim=1)[0] 
            prob_id= int(torch.argmax(prob).item())
            conf = float(prob[prob_id].item())

            images.append(img_t.cpu())
            gts.append(label)
            preds.append(prob_id)
            confs.append(conf)

    # --- 畫出預測結果 ---
    os.makedirs(OUT_DIR,exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE,constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS,COLS,i+1)
        # 把圖片從正規化的狀態恢復回原始可顯示樣子
        img=denorm_img(images[i])



        # | `img.squeeze()`           | .squeeze() 會移除 tensor 或陣列中「維度大小為 1」的維度。 | (1, 28, 28)shape → `[28, 28]` |
        # | `imshow()`                | 顯示圖片          | 顯示為灰階圖             |
        # | `cmap="gray"`             | 用灰階顏色        | 黑白圖像               |
        # | `interpolation="nearest"` | 不做模糊插值      | 保留像素邊界             |

        ax.imshow(img.squeeze(),cmap="gray",interpolation="nearest")
        ax.axis("off")

        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt = f"{confs[i]:.3f}"

        # 標題：真實 vs 預測 + 信心值
        title = f"真:{gt_name}\n預:{pred_name}({conf_txt})"
        ax.set_title(title,color="green",fontsize=TITLE_FONTSIZE,pad=6)

    fig.suptitle("Fashion-MNIST 測試集預測結果", fontsize=16, y=0.98)

    # 輸出結果圖 test_grid.png

    out_path=Path(OUT_DIR)/"test_grid.png"

    # bbox_inches="tight" =>它會幫你把圖片裁到剛好包住所有可視內容，不多不寡。
    fig.savefig(out_path,dpi=180,bbox_inches="tight")
    plt.close(fig)
    # resolve() 的作用是把路徑「解析成完整的絕對路徑」。
    print(f"[OK] 圖片已輸出: {out_path.resolve()}")

if __name__ == "__main__":
    main()

