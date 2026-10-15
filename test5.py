import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体防止中文乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")

CLASS_NAMES = ["横纹", "竖纹", "对角线", "棋盘"]

class PatternDataset(Dataset):
    def __init__(self, n_samples=20000, image_size=28, n_classes=4):
        super().__init__()
        self.n = n_samples
        self.image_size = image_size
        self.n_classes = n_classes

        labels = torch.arange(n_classes).repeat_interleave(n_samples // n_classes)
        if labels.size(0) < n_samples:
            extra = n_samples - labels.size(0)
            labels = torch.cat([labels, labels[:extra]], dim=0)
        self.labels = labels[torch.randperm(n_samples)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = int(self.labels[idx].item())
        image = self._generate_pattern(label)
        return image, label

    def _generate_pattern(self, label):
        size = self.image_size
        image = torch.zeros(size, size, dtype=torch.float32)

        if label == 0:
            # 横向条纹
            count = torch.randint(1, 4, (1,)).item()
            for _ in range(count):
                thickness = torch.randint(1, 4, (1,)).item()
                row = torch.randint(0, size - thickness + 1, (1,)).item()
                image[row:row + thickness, :] = 1.0
        elif label == 1:
            # 竖向条纹
            count = torch.randint(1, 4, (1,)).item()
            for _ in range(count):
                thickness = torch.randint(1, 4, (1,)).item()
                col = torch.randint(0, size - thickness + 1, (1,)).item()
                image[:, col:col + thickness] = 1.0
        elif label == 2:
            # 对角线
            thickness = torch.randint(1, 4, (1,)).item()
            direction = torch.randint(0, 2, (1,)).item()
            base = torch.eye(size, size)
            for offset in range(-thickness + 1, thickness):
                diag = torch.roll(base, shifts=offset, dims=1)
                if direction == 1:
                    diag = torch.flip(diag, dims=[1])
                image += diag
            image = torch.roll(image, shifts=torch.randint(-4, 5, (1,)).item(), dims=1)
        else:
            # 棋盘格
            tile = torch.randint(2, 6, (1,)).item()
            for i in range(0, size, tile):
                for j in range(0, size, tile):
                    if ((i // tile) + (j // tile)) % 2 == 0:
                        image[i:i + tile, j:j + tile] = 1.0

        noise = torch.randn(size, size) * 0.25
        image = image + noise
        image = torch.clamp(image, 0.0, 1.0)
        return image.unsqueeze(0)


def plot_sample_images(dataset, num_samples=12):
    indices = torch.randperm(len(dataset))[:num_samples]
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
    for idx, sample_idx in enumerate(indices):
        image, label = dataset[int(sample_idx)]
        plt.subplot(nrows, ncols, idx + 1)
        plt.imshow(image.squeeze(0).numpy(), cmap="gray")
        plt.title(CLASS_NAMES[label])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


class SimpleNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


batch_size = 64
train_ds = PatternDataset(n_samples=2000, image_size=28, n_classes=len(CLASS_NAMES))
test_ds = PatternDataset(n_samples=600, image_size=28, n_classes=len(CLASS_NAMES))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = SimpleNet(n_classes=len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

train_loss_hist = []
test_acc_hist = []

print("展示部分随机图案样本...")
plot_sample_images(train_ds)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            out = model(x)
            _, pred = torch.max(out, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).long()
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_loss_hist.append(avg_loss)

    acc = evaluate(model, test_loader)
    test_acc_hist.append(acc)

    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_hist, marker='o')
plt.title("训练损失")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_acc_hist, marker='o')
plt.title("测试准确率")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("训练完成，展示测试集预测示例...")
model.eval()
plt.figure(figsize=(10, 5))
for i in range(12):
    image, label = test_ds[i]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        pred_label = torch.argmax(pred, dim=1).item()
    plt.subplot(3, 4, i + 1)
    plt.imshow(image.squeeze(0).numpy(), cmap="gray")
    plt.title(f"真:{CLASS_NAMES[label]}\n预测:{CLASS_NAMES[pred_label]}")
    plt.axis("off")
plt.tight_layout()
plt.show()