# 彩色圖像的 mean/std 計算方式
imgs = torch.stack([img for img, _ in dataset])  # shape: [N, 3, H, W]
print(imgs.shape)  # 例如 [50000, 3, 32, 32]


# 🔹2️⃣ 對每個「通道」分別計算 mean & std
# 這裡 dim=(0, 2, 3) 的意思是：
# dim 0 → 所有圖片 (50000 張)
# dim 1 → 通道 (R,G,B)
# dim 2 → 高度 (32)
# dim 3 → 寬度 (32)
# 意思是：
# 對維度 0、2、3 進行平均，只保留維度 1（也就是 RGB 通道）。

mean = imgs.mean(dim=(0, 2, 3))
std = imgs.std(dim=(0, 2, 3))

print("mean:", mean)
print("std:", std)


# 結果會是三個值：

mean: tensor([0.4914, 0.4822, 0.4465])
std:  tensor([0.2470, 0.2435, 0.2616])


# 彩色圖片有 R、G、B 三個通道，
# 每個通道要有各自的 mean 和 std。
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # R, G, B
        std=(0.2470, 0.2435, 0.2616)
    )
])


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # ✅ 改成輸入 3 通道（RGB）
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3, 32, 32) → (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # → (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32×32 → 16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16×16 → 8×8

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # ✅ 改這裡 (影像尺寸不同)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x