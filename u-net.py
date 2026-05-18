import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import numpy as np
from tqdm import tqdm
import random
import os
from PIL import Image
import wandb  # 使用 wandb 进行可视化 

# =========================== 0. 基础配置 ===========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================== 1. 数据加载与预处理 = [cite: 18] ===========================
class SegmentationTransform:
    """同时应用于图像和分割标签的变换 [cite: 18]"""
    def __init__(self, img_size=(256, 256), augment=False):
        self.img_size = img_size
        self.augment = augment

    def __call__(self, img, mask):
        # 调整大小
        img = TF.resize(img, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # 随机水平翻转（增强）
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Oxford-IIIT Pet Mask: 1→前景, 2→背景, 3→边缘 [cite: 19]
        # 转换为索引: 0→pet, 1→background, 2→outline [cite: 18]
        mask = mask - 1
        mask = torch.clamp(mask, 0, 2)
        return img, mask

def load_oxford_pet_dataset(data_dir='./data', batch_size=16):
    transform_train = SegmentationTransform(augment=True)
    transform_val = SegmentationTransform(augment=False)

    train_data = datasets.OxfordIIITPet(root=data_dir, split='trainval', target_types='segmentation', download=True)
    test_data = datasets.OxfordIIITPet(root=data_dir, split='test', target_types='segmentation', download=True)

    class PetDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            img, mask = self.dataset[idx]
            return self.transform(img, mask)

    train_loader = DataLoader(PetDataset(train_data, transform_train), batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(PetDataset(test_data, transform_val), batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, val_loader

# =========================== 2. 从零实现 U-Net =  ===========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    """上采样并包含跳跃连接 (Skip Connection) [cite: 17]"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 特征拼接 [cite: 17]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """标准 U-Net 结构 [cite: 17]"""
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# =========================== 3. 损失函数工程 =  ===========================
class DiceLoss(nn.Module):
    """手动实现多分类 Dice Loss """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_all = 0
        for i in range(num_classes):
            if_intersect = (probs[:, i] * targets_one_hot[:, i]).sum()
            if_union = probs[:, i].sum() + targets_one_hot[:, i].sum()
            dice_all += (2. * if_intersect + self.smooth) / (if_union + self.smooth)
        
        return 1 - (dice_all / num_classes)

class CombinedLoss(nn.Module):
    """CE + Dice 组合损失 [cite: 23]"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)

# =========================== 4. 评估指标 =  ===========================
def get_metrics(pred, target):
    pred_idx = torch.argmax(pred, dim=1)
    acc = (pred_idx == target).float().mean()
    # 简易 mIoU 计算
    iou = 0
    for i in range(3):
        inter = ((pred_idx == i) & (target == i)).sum().float()
        union = ((pred_idx == i) | (target == i)).sum().float()
        iou += (inter + 1e-6) / (union + 1e-6)
    return acc.item(), (iou / 3).item()

# =========================== 5. 训练主逻辑 =  ===========================
def run_experiment(exp_name, criterion, train_loader, val_loader, epochs=30):
    wandb.init(project="Pet-Segmentation-Comparison", name=exp_name, config={"epochs": epochs, "lr": 1e-4})
    
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_iou = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"{exp_name} Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss, val_acc, val_iou = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                acc, iou = get_metrics(outputs, masks)
                val_acc += acc
                val_iou += iou
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        # 终端打印指标
        print(f"{exp_name} | Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f} | mIoU: {avg_val_iou:.4f}")

        # 记录到 wandb
        wandb.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/Accuracy": avg_val_acc,
            "val/mIoU": avg_val_iou
        })

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"best_{exp_name}.pth")

    wandb.finish()
def main():
    train_loader, val_loader = load_oxford_pet_dataset(batch_size=16)

    # 实验对比 [cite: 21, 22, 23]
    experiments = [
        ("CrossEntropy_Only", nn.CrossEntropyLoss()),
        ("DiceLoss_Only", DiceLoss()),
        ("Combined_CE_Dice", CombinedLoss())
    ]

    for name, criterion in experiments:
        run_experiment(name, criterion, train_loader, val_loader)

if __name__ == "__main__":
    main()