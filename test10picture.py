import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from typing import List, Tuple
import struct

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从BigHomework.py复制必要的类和函数
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

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

def load_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载测试数据集"""
    images_file = os.path.join(data_dir, 'train-images.idx3-ubyte')
    labels_file = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    
    images = read_idx(images_file)
    labels = read_idx(labels_file)
    
    return images, labels

def select_random_samples(images: np.ndarray, labels: np.ndarray, n_samples: int = 10) -> Tuple[List[np.ndarray], List[int]]:
    """随机选择n_samples张图片"""
    total_samples = len(images)
    selected_indices = random.sample(range(total_samples), n_samples)
    
    selected_images = [images[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    
    return selected_images, selected_labels

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """预处理图片"""
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    # 转换为tensor并添加批次和通道维度
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    image_tensor = transform(image_tensor)
    
    return image_tensor

def predict_images(model: nn.Module, images: List[np.ndarray], device: torch.device) -> List[Tuple[int, float]]:
    """预测图片"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for image in images:
            image_tensor = preprocess_image(image)
            image_tensor = image_tensor.to(device)
            
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            predictions.append((prediction, confidence))
    
    return predictions

def visualize_results(images: List[np.ndarray], labels: List[int], predictions: List[Tuple[int, float]], 
                     total_time: float, accuracy: float):
    """可视化测试结果"""
    n_samples = len(images)
    fig = plt.figure(figsize=(15, 10))
    
    # 创建一个大标题
    fig.suptitle(f'随机测试结果\n总用时: {total_time:.2f}秒, 准确率: {accuracy:.2f}%', fontsize=16)
    
    # 显示每张图片和预测结果
    for i in range(n_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        pred, conf = predictions[i]
        color = 'green' if pred == labels[i] else 'red'
        plt.title(f'真实值: {labels[i]}\n预测值: {pred}\n置信度: {conf:.2f}%', 
                 color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        print("开始运行随机测试程序...")
        
        # 设置随机种子
        random.seed(42)
        torch.manual_seed(42)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {device}')
        
        # 加载模型
        model_path = 'saved_models/mnist_model_best.pth'
        if not os.path.exists(model_path):
            model_path = 'saved_models/mnist_model.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("未找到模型文件，请先运行训练程序")
        
        print(f"加载模型: {model_path}")
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(model_path))
        
        # 加载数据集
        data_dir = r"D:\AAAinformation\Course\TheoryAndApplicationOfArtficalIntelligence\MNIST\MNIST"
        print("加载数据集...")
        images, labels = load_test_data(data_dir)
        
        # 随机选择10张图片
        print("随机选择10张图片...")
        selected_images, selected_labels = select_random_samples(images, labels)
        
        # 开始计时
        start_time = time.time()
        
        # 预测
        print("开始预测...")
        predictions = predict_images(model, selected_images, device)
        
        # 计算用时
        total_time = time.time() - start_time
        
        # 计算准确率
        correct = sum(1 for (pred, _), label in zip(predictions, selected_labels) if pred == label)
        accuracy = (correct / len(selected_labels)) * 100
        
        print(f"\n测试完成!")
        print(f"总用时: {total_time:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        
        # 可视化结果
        print("\n显示测试结果...")
        visualize_results(selected_images, selected_labels, predictions, total_time, accuracy)
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 