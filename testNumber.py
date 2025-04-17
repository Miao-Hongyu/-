import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
from typing import List, Tuple

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

def segment_digits(image_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """分割图片中的连续数字"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 保存原始图片用于显示
    original_image = image.copy()
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 图像预处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 进行形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 过滤和排序数字区域
    digit_regions = []
    min_area = 100  # 最小区域面积
    min_aspect_ratio = 0.2  # 最小宽高比
    max_aspect_ratio = 3.0  # 最大宽高比
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        if (area > min_area and 
            min_aspect_ratio < aspect_ratio < max_aspect_ratio):
            # 扩大边界框以包含完整数字
            margin = 4
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(binary.shape[1] - x, w + 2 * margin)
            h = min(binary.shape[0] - y, h + 2 * margin)
            digit_regions.append((x, y, w, h))
    
    # 按照x坐标排序，从左到右
    digit_regions.sort(key=lambda x: x[0])
    
    # 提取每个数字
    digits = []
    for x, y, w, h in digit_regions:
        # 在原始图片上画框（用于显示）
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 提取数字区域
        digit = binary[y:y+h, x:x+w]
        
        # 添加边距
        padding = int(min(w, h) * 0.2)
        digit = cv2.copyMakeBorder(
            digit, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 调整为正方形
        size = max(digit.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - digit.shape[1]) // 2
        y_offset = (size - digit.shape[0]) // 2
        square[y_offset:y_offset+digit.shape[0], 
               x_offset:x_offset+digit.shape[1]] = digit
        
        digits.append(square)
    
    return digits, original_image

def preprocess_digit(digit_image: np.ndarray) -> torch.Tensor:
    """预处理单个数字图像"""
    # 调整大小为28x28
    digit_resized = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 应用高斯模糊减少噪声
    digit_resized = cv2.GaussianBlur(digit_resized, (3, 3), 0)
    
    # 归一化
    digit_normalized = digit_resized.astype(np.float32) / 255.0
    
    # 确保黑白颜色正确
    if digit_normalized.mean() > 0.5:
        digit_normalized = 1 - digit_normalized
    
    # 增强对比度
    digit_normalized = np.clip((digit_normalized - digit_normalized.mean()) * 1.5 + 0.5, 0, 1)
    
    # 转换为tensor并添加批次和通道维度
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = torch.FloatTensor(digit_normalized).unsqueeze(0).unsqueeze(0)
    image_tensor = transform(image_tensor)
    
    return image_tensor

def predict_digits(model: nn.Module, digits: List[np.ndarray], device: torch.device) -> List[Tuple[int, float]]:
    """预测多个数字"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for digit in digits:
            image_tensor = preprocess_digit(digit)
            image_tensor = image_tensor.to(device)
            
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            predictions.append((prediction, confidence))
    
    return predictions

def visualize_results(original_image: np.ndarray, digits: List[np.ndarray], 
                     predictions: List[Tuple[int, float]], total_time: float):
    """可视化测试结果"""
    n_digits = len(digits)
    fig = plt.figure(figsize=(15, 8))
    
    # 创建一个大标题
    predicted_number = ''.join([str(pred[0]) for pred in predictions])
    confidence_avg = sum(pred[1] for pred in predictions) / len(predictions) * 100
    fig.suptitle(f'手写数字识别结果\n预测数字: {predicted_number}\n'
                 f'平均置信度: {confidence_avg:.2f}%\n'
                 f'识别用时: {total_time:.2f}秒', fontsize=16)
    
    # 显示原始图片（带标注框）
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('原始图片（绿框表示检测到的数字）')
    plt.axis('off')
    
    # 显示每个分割的数字和预测结果
    for i in range(n_digits):
        plt.subplot(2, n_digits, n_digits + i + 1)
        plt.imshow(digits[i], cmap='gray')
        pred, conf = predictions[i]
        plt.title(f'预测: {pred}\n置信度: {conf:.2f}%')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        print("开始运行手写数字识别程序...")
        
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
        
        # 设置图片路径
        image_dir = 'test_numbers'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"创建测试图片目录: {os.path.abspath(image_dir)}")
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"警告：在 {image_dir} 目录中没有找到图片文件")
            print("请将手写数字图片放入该目录")
            return
        
        # 处理每张图片
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"\n处理图片: {image_file}")
            
            # 开始计时
            start_time = time.time()
            
            # 分割数字
            digits, original_image = segment_digits(image_path)
            if not digits:
                print("未检测到任何数字！")
                continue
            
            # 预测
            print("开始预测...")
            predictions = predict_digits(model, digits, device)
            
            # 计算用时
            total_time = time.time() - start_time
            
            # 显示结果
            predicted_number = ''.join([str(pred[0]) for pred in predictions])
            confidence_avg = sum(pred[1] for pred in predictions) / len(predictions) * 100
            
            print(f"预测结果: {predicted_number}")
            print(f"平均置信度: {confidence_avg:.2f}%")
            print(f"识别用时: {total_time:.2f}秒")
            
            # 可视化结果
            print("显示识别结果...")
            visualize_results(original_image, digits, predictions, total_time)
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 