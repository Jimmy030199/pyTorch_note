from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torchvision import datasets , transforms
from cnn import CNN
import torch.nn.functional as F
import os


DATA_DIR="data"
OUT_DIR="result"
CKPT_PATH=Path(OUT_DIR) / "best_cnn.pth"
ROWS=2
COLS=5
SEED=42
MEAN=(0.4914, 0.4822, 0.4465)
STD=(0.2470, 0.2435, 0.2616)
FIGSIZE=(16,8)
TITLE_SIZE=11


ZH_FONT_CANDIDATES =[
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]

CLASS_NAMES =['airplane', 'automobile', 'bird', 'cat', 'deer', 
   'dog', 'frog', 'horse', 'ship', 'truck'
   ]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_zh_font():
    for path in ZH_FONT_CANDIDATES:
        if Path(path).exists():
            plt.rcParams["font.sans-serif"]=[font_manager.FontProperties(fname=path).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO]已啟用中文字體:{path}")
            return
    print("[WARN]找不到中文字體，中文可能無法正常顯示。")

def denorm_img(x):
    # 這裡的 .view(3,1,1) 是關鍵：
    # 把 [3] 變成 [3,1,1]
    #  [[[a]],
    #  [[b]],
    #  [[c]]]
    # .view(3,32,32) 能跑沒錯，但會「真的複製出整張 32×32 的矩陣」；
    # .view(3,1,1) 才是聰明做法——它讓 PyTorch 自動在運算時「虛擬展開」而不浪費空間。
    # PyTorch 就能廣播到 [3,H,W]，每個通道自動套對應的 mean/std。
    mean_view = torch.tensor(MEAN, device=x.device).view(3, 1, 1)
    std_view = torch.tensor(STD, device=x.device).view(3, 1, 1)

    x = x * std_view + mean_view
    return x.clamp(0,1)


# ===============

def main():
    set_seed(SEED)
    set_zh_font()

    device=torch.device("cpu")

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN,STD)
        ]
    )

    test_set = datasets.CIFAR10(
        root = DATA_DIR,
        train=False,
        download=True,
        transform=tfm
    )

    model = CNN(num_classes=10).to(device)

    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH,map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] 已載入模型權重:{CKPT_PATH}")
    else:
        print(f"[WARN] 模型檔案不存在: {CKPT_PATH}，使用隨機初始化權重")
    
    model.eval()

    N = COLS * ROWS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices= indices[:N]

    images=[]
    gts=[]
    preds=[]
    confs=[]

    with torch.no_grad():
        for idx in indices:
            img_t,label = test_set[idx]
            # 增加 batch 維度 才可以是dataloader的shape
            # unsqueeze(0) → 增加 batch 維度 [1, 1, 32, 32]
            # logits 的 shape 是 [batch_size, num_classes] → [1, 10]
            logits = model(img_t.unsqueeze(0).to(device))
            #[0]因為 長[[0.7054, 0.2369, 0.0577.......]]
            prob =F.softmax(logits,dim=1)[0]
            prob_id = int(torch.argmax(prob).item())
            conf = float(prob[prob_id].item()) 

            images.append(img_t.cpu())
            gts.append(label)
            preds.append(prob_id)
            confs.append(conf)
    
    os.makedirs(OUT_DIR,exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE,constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS,COLS,i+1)
        img = denorm_img(images[i])

        #為什麼要 .permute(1, 2, 0) ？
        # PyTorch 的影像張量是：
        # [通道數, 高, 寬] = [C, H, W]
        # 例如：
        # img.shape = [3, 32, 32]
        # 但 Matplotlib 需要的是：
        # [高, 寬, 通道數] = [H, W, C]
        # 所以要把軸順序換一下：
        # img = img.permute(1, 2, 0)
        ax.imshow(img.permute(1, 2, 0),interpolation="nearest")
        ax.axis("off")

        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt=f"{confs[i]:.3f}"

        title = f"真:{gt_name} \n預:{pred_name}({conf_txt})"
        ax.set_title(title,color="green",fontsize=TITLE_SIZE,pad=6)
    fig.suptitle("CIFAR10 測試集預測結果", fontsize=16, y=0.98)
    out_path = Path(OUT_DIR) /"test_grid.png"

    fig.savefig(out_path,dpi=180,bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 圖片已輸出: {out_path.resolve()}")

if __name__ == "__main__":
    main()