import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import struct
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
# 设置CUDA随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

class CustomMNIST(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        if train:
            images_file = os.path.join(data_dir, 'train-images.idx3-ubyte')
            labels_file = os.path.join(data_dir, 'train-labels.idx1-ubyte')
        else:
            images_file = os.path.join(data_dir, 't10k-images.idx3-ubyte')
            labels_file = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
        
        self.images = read_idx(images_file)
        self.labels = read_idx(labels_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为PIL Image
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# CNN模型定义
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 数据加载和预处理函数
def load_data(data_path, batch_size=1024):
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = CustomMNIST(
        data_dir=data_path,
        train=True,
        transform=transform
    )
    
    test_dataset = CustomMNIST(
        data_dir=data_path,
        train=False,
        transform=transform
    )
    
    # 设置数据加载器的工作进程数
    num_workers = 8 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader

# 训练函数
def train_model(model, train_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 启用 CUDA 优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    train_losses = []
    train_accuracies = []
    best_accuracy = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印训练设备信息
    print(f'训练设备: {device}')
    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'初始GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB')
        print(f'初始GPU显存缓存: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 数据增强
            if epoch > 5:
                # 添加随机噪声
                noise = torch.randn_like(images) * 0.1
                images = images + noise
                # 随机对比度调整
                contrast = torch.rand(1).item() * 0.5 + 0.7  # 0.7-1.2
                images = images * contrast
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        # 更新学习率
        scheduler.step(epoch_loss)
        
        # 检查学习率是否发生变化
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f'学习率从 {current_lr:.6f} 调整为 {new_lr:.6f}')
            current_lr = new_lr
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f'轮次 [{epoch+1}/{num_epochs}], 损失: {epoch_loss:.4f}, 准确率: {epoch_accuracy:.2f}%, 学习率: {current_lr:.6f}')
        
        if device.type == 'cuda':
            print(f'当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB')
            print(f'当前GPU显存缓存: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB')
        
        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), 'saved_models/mnist_model_best.pth')
    
    return train_losses, train_accuracies

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

# 主函数
def main():
    try:
        # 创建模型保存目录
        model_save_dir = 'saved_models'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            print(f"创建模型保存目录: {os.path.abspath(model_save_dir)}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        if device.type == 'cuda':
            print(f'可用GPU数量: {torch.cuda.device_count()}')
            print(f'GPU型号: {torch.cuda.get_device_name(0)}')
        
        # 数据路径
        data_path = r"D:\AAAinformation\Course\TheoryAndApplicationOfArtficalIntelligence\MNIST\MNIST"
        
        # 确保数据目录存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据目录不存在: {data_path}")
            
        # 加载数据
        train_loader, test_loader = load_data(data_path)
        
        # 创建模型
        model = ConvNet().to(device)
        
        # 如果有多个GPU，使用数据并行
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU训练")
            model = nn.DataParallel(model)
        
        # 训练模型
        num_epochs = 30  # 增加训练轮数
        train_losses, train_accuracies = train_model(model, train_loader, num_epochs, device)
        
        # 测试模型
        test_accuracy = test_model(model, test_loader, device)
        
        # 保存模型
        model_save_path = os.path.join(model_save_dir, 'mnist_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存到: {os.path.abspath(model_save_path)}")
        
        # 绘制损失曲线和准确率曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, 'b-')
        plt.title('训练损失随轮次的变化')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_accuracies, 'r-')
        plt.title('训练准确率随轮次的变化')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()
