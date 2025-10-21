import argparse
import os
from torchvision import datasets,transforms
import torch
from torch.utils.data import random_split,DataLoader
from cnn import CNN
import torch.nn as nn
import torch.optim as optim
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data')
    parser.add_argument('--out_dir',type=str,default='result')
    parser.add_argument('--epoch',type=int,default=15)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=5e-4)

    args=parser.parse_args()
    print("args",args)

    os.makedirs(args.data_dir,exist_ok=True)
    os.makedirs(args.out_dir,exist_ok=True)

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
        )
    
    print(dataset.classes) 
    #['airplane', 'automobile', 'bird', 
    # 'cat', 'deer', 'dog', 
    # 'frog', 'horse', 'ship', 
    # 'truck']
    
    imgs = torch.stack([img for img,_ in dataset])
    print(imgs.shape) #[50000, 3, 32, 32]

    dimSelect=(0,2,3)
    mean = imgs.mean(dimSelect)
    std = imgs.std(dimSelect)

    print("mean",mean) #[0.4914, 0.4822, 0.4465]
    print("std",std) #[0.2470, 0.2435, 0.2616]

    tfm= transforms.Compose([
        transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))                   
        ])
    full_train = datasets.CIFAR10(
        root = args.data_dir,
        train=True,
        download=True,
        transform=tfm
    )
    img,label= full_train[0]
    print('no.1 img shape',img.shape)
    # labels（CIFAR-10）本身就是整數標籤，
    print('no.1 label',label)

    train_ratio=0.8

    train_len = int(len(full_train) * train_ratio)
    val_len = len(full_train) - train_len

    train_set,val_set = random_split(full_train,[train_len,val_len])

    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size = args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device('cpu')
    model = CNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)


    loss_csv = os.path.join(args.out_dir,'loss.csv')
    with open(loss_csv,'w',encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
    
    best_val_loss = float('inf')
    best_path= os.path.join(args.out_dir,'best_cnn.pth')

    print('model 的架構',model)


    for epoch in range(1,args.epoch+1):
        t0 = time.time()
        model.train()
        running_loss,running_acc,n=0.0,0.0,0
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # forward
            logits = model(images)
            loss = criterion(logits,labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and acc 數字紀錄
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n+=batch_size
            running_acc += (logits.argmax(1) == labels).float().sum().item()

        train_loss = running_loss / n
        train_acc = running_acc / n

        # ---valicate---
        model.eval()
        running_val_loss,running_val_acc,m=0.0,0.0,0

        with  torch.no_grad():
            for images,labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()

                logits = model(images)
                loss = criterion(logits,labels)

                batch_size = labels.size(0)
                running_val_loss += loss.item() * batch_size
                m+=batch_size
                running_val_acc += (logits.argmax(1) == labels).float().sum().item()
            
            val_loss =running_val_loss / m
            val_acc =running_val_acc / m
    
        with open(loss_csv,'a',encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")

        dt=time.time() - t0
        print(
            f"[Epoch {epoch:02d}]"
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} " 
            f"({dt:.1f}s)"
            )
        
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            torch.save({
                'epoch':epoch,
                'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'val_loss':val_loss,
                'val_acc':val_acc,
            },best_path)
            print(f"-->Saved new best to {best_path}")










if __name__ == "__main__":
    main()
