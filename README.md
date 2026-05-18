# 实现 U-Net 与损失函数工程（Oxford‑IIIT Pet）

本项目从零搭建 U‑Net 语义分割网络，不使用任何预训练权重。在 Oxford‑IIIT Pet 数据集上对比 交叉熵损失 (CE)、Dice Loss 以及 CE + Dice 组合损失 在像素级分割任务中的表现，评估指标为 mIoU 和像素准确率。

## 环境配置

数据集准备
代码会自动下载 Oxford‑IIIT Pet Dataset（包括图像和分割标注）,但有时候会超时失败，所以建议自行下载至./data目录下。

## 训练与对比实验
运行 u_net.py 将依次执行三种损失函数的训练：

CrossEntropy_Only – 仅交叉熵损失

DiceLoss_Only – 仅手动实现的 Dice Loss

Combined_CE_Dice – 交叉熵 + Dice 组合损失

## 输出结果

依次训练三个模型，每个训练完成后保存最佳模型权重（基于验证集 mIoU）

实时输出训练损失、验证准确率、验证 mIoU

使用 wandb 记录训练曲线
