import os
import argparse
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
import torch
from cnn import CNN
import torch.nn as nn
import torch.optim as optim
import time



def main():
    # [Step 1] — 命令列參數解析
    # 目的：讓使用者能在執行程式時，自訂路徑、訓練回合數、批次大小、學習率等參數。
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',type=str,default='data',help='資料下載/讀取目錄')
    parser.add_argument('--out_dir',type=str,default='result',help='訓練輸出目錄')
    parser.add_argument('--epoch',type=int,default=10,help='訓練回合數')
    parser.add_argument('--batch_size',type=int,default=128,help='批次大小')
    parser.add_argument('--lr',type=float,default=1e-3,help='學習率')
    
    args=parser.parse_args()
    # print("args",args)

    os.makedirs(args.data_dir,exist_ok=True)
    os.makedirs(args.out_dir,exist_ok=True)
    # print(f"data_dir={args.data_dir},out_dir={args.out_dir}")


    # [Step 2] — 資料載入與前處理
    # 目的：下載並轉換 Fashion-MNIST 資料集，拆分為訓練集與驗證集。
    # 最佳實踐是：訓練和測試都用相同 mean/std
    # 前處理
    
    # 先下載資料集
    # dataset 每筆都是 (影像 tensor, 標籤 int)
    dataset = datasets.FashionMNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
        )

    # 將所有影像堆疊成一個 tensor
    # 這段會做的事情是：
    # 逐筆取出 dataset 中的 (img, label)
    # 只保留影像 img，不取標籤。忽略 _（底線代表「我不需要這個變數」）
    # torch.stack([...]) 會把所有影像堆疊起來形成一個大 tensor。
    # 假設每張影像 shape 為 [1, 28, 28]（單通道 28x28 灰階圖像），
    # 而 dataset 有 60,000 張圖，
    # 那堆疊後 imgs 的 shape 變成 [60000, 1, 28, 28]。
    imgs = torch.stack([img for img, _ in dataset])  
    print(imgs.shape)# shape: [60000, 1, 28, 28]

    # 計算 mean & std
    mean = imgs.mean().item() #這行計算「整個 tensor 所有像素的平均值」。
    std = imgs.std().item()

    print("mean =", mean)
    print("std =", std)
    
    # transforms.Normalize((0.2861,), (0.3530,))
    # 是針對 每個「通道 channel」 做正規化，
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2861,),(0.3530,))
        ])
    # print('tfm',tfm)
    full_train = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=tfm
    )
    # 取出第一張圖與標籤
    img, label = full_train[0]
    print('第一筆資料的影像shape',img.shape)
    print('第一筆資料的label',label)


    # 2️⃣ 設定比例
    train_ratio = 0.8
    val_ratio = 0.2

    # 3️⃣ 計算實際長度（筆數）
    train_len = int(len(full_train) * train_ratio)
    val_len = len(full_train) - train_len

    # 4️⃣ 隨機切分
    train_set, val_set = random_split(full_train, [train_len, val_len])




    # [Step 3] — 建立 DataLoader
    # 目的：以批次方式輸入資料，方便 GPU 平行處理。
    train_loader= DataLoader(
        train_set,
        batch_size= args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader= DataLoader(
        val_set,
        batch_size= args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # [Step 4] — 建立模型與訓練設定
    # 目的：初始化模型、損失函數與優化器。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =CNN(num_classes=10).to(device)

    criterion =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    print("Loss / Optimizer 準備完成")


    #[ Step 5] — 模型訓練與驗證迴圈及訓練紀錄與輸出
    # 目的：反覆訓練模型、驗證效能、記錄結果。將訓練過程寫入 CSV，方便後續繪圖或分析。

    # [放在 epoch 迴圈外]
    loss_csv=os.path.join(args.out_dir,'loss.csv')
    with open(loss_csv,'w',encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
    # 儲存最佳模型
    best_val_loss=float('inf') #初始化「最佳驗證損失」為無限大 (infinity)
    best_path = os.path.join(args.out_dir,'best_cnn.pth')  
    # -------------------------------------
    for epoch in range(1, args.epoch +1):
        t0=time.time()
        model.train() #切換到訓練模式 (啟用 dropout, BN)
        running_loss,running_acc,n=0.0,0.0,0 # n代表計算目前累積處理的樣本總數
        for images,labels in train_loader: #images labels都是一整個batch
            images=images.to(device)
            labels=labels.to(device).long() # CrossEntropyLoss 需要 int64
            
            # Forward

            # 模型輸出 [batch, 10] 輸出的「原始分數」logits 是模型最後一層（通常是全連接層 Linear）的原始輸出值，
            # 還 沒經過 softmax 或 sigmoid 等歸一化處理
            logits=model(images) 

            # 注意：CrossEntropyLoss 的預設行為是 回傳這個 batch 的平均 loss（不是總和）
            # Cross-Entropy Loss 中 PyTorch 會自動幫你做 softmax，不用自己加
            loss=criterion(logits,labels) # 計算損失 (未 softmax) 
            
            # Backward
            optimizer.zero_grad()
            loss.backward() #會影響模型而running_loss只是記錄用
            optimizer.step()

            # 累積損失 (乘 batch size，換算成總和)
            batch_size =labels.size(0) #取得這個 batch 的大小
            running_loss += loss.item()*batch_size #running_loss是「紀錄」用
            n+=batch_size
            
            #計算這一批（batch）中有幾筆分類正確
            # argmax=在「類別維度」上找出分數最高的索引 如:tensor([0, 1, 0])
            # logits.argmax(1) == labels 如:tensor([True, False, True, True])
            # .item() 用意回傳一個float(非tensor格式)
            # argmax(1) 中的1 是 dim=1
            # 維度 0 是 batch（樣本數）
            # 維度 1 是 類別分數 (class logits)
            running_acc += (logits.argmax(1) == labels).float().sum().item()

        # 每做一個epoch:
        # 計算平均訓練損失 及 平均正確率
        train_loss = running_loss / n
        train_acc =running_acc/ n

        # --- validate ---
        model.eval()
        val_loss,val_acc,m =0.0,0.0,0

        with torch.no_grad():
            for images,labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()

                logits=model(images)
                loss =criterion(logits,labels)


                batch_size=labels.size(0)
                val_loss +=loss.item()*batch_size
                val_acc += (logits.argmax(1)==labels).float().sum().item()
                m+=batch_size
            # 每做一個epoch:
            # 計算平均訓練損失 及 平均正確率
            val_loss=val_loss/m
            val_acc=val_acc/m

    
        # 訓練紀錄與輸出(跟資料的for迴圈同層)
        # 目的：將訓練過程寫入 CSV，方便後續繪圖或分析。  
        # 將結果寫入 csv 檔
        with open(loss_csv,'a',encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")
        
        # 顯示目前 epoch 的結果
        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"({dt:.1f}s) "
              )
        # 若 validation loss 下降，則儲存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,  # 記錄目前是第幾個 epoch
                'model_state': model.state_dict(),  # 模型所有權重參數
                'optimizer_state': optimizer.state_dict(),  # 優化器的狀態（例如 Adam 的 momentum）
                'val_loss': val_loss,  # 這次 epoch 的驗證損失
                'val_acc': val_acc,  # 這次 epoch 的驗證正確率
            },best_path)
            print(f"-->Saved new best to {best_path}")


if __name__ =="__main__":
    main()