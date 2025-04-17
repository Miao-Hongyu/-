## 功能特点
- ✨ 支持单个数字识别
- 📝 支持连续数字序列识别
- 🎯 高准确率 (>99% on MNIST)
- ⚡ 实时处理
- 📊 结果可视化
- 🔄 批量处理支持

## 系统要求
- Python 3.8+
- CUDA 11.0+ (GPU加速，可选)
- 8GB+ RAM
- Windows 10/11 或 Linux

## 依赖库
```bash
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
opencv-python==4.8.0
matplotlib==3.7.1
```

## 使用说明

### 1. 训练模型

```bash
python BigHomework.py
```

训练完成后，模型将保存在 `saved_models` 目录下。

### 2. 随机测试

```bash
python test10picture.py
```

从训练集随机选择10张图片进行测试。

### 3. 识别手写数字

```bash
python testNumber.py
```

### 目录结构

```
├── BigHomework.py          # 训练程序
├── test10picture.py        # 随机测试程序
├── testNumber.py           # 手写数字识别程序
├── saved_models/           # 保存的模型
│   └── mnist_model.pth
├── test_numbers/          # 测试图片目录
│   ├── test1.jpg
│   └── test2.png
└── requirements.txt       # 依赖库列表
```

## 测试图片要求

1. 格式支持
   - PNG
   - JPG/JPEG

2. 图片要求
   - 白底黑字或黑底白字
   - 数字清晰可见
   - 数字间有适当间距
   - 建议分辨率：至少300x300像素

3. 存放位置
   - 将测试图片放入 `test_numbers` 目录

## 性能指标

- 训练集准确率: 99.7%+
- 测试集准确率: 99.4%+
- 实际手写数字识别准确率: 94%+
- 单张图片处理时间: <0.5秒
- GPU显存占用: ~430MB

## 注意事项

1. 训练相关
   - 确保有足够的磁盘空间（至少5GB）
   - GPU训练时确保显存充足
   - 训练过程中不要中断程序

2. 测试相关
   - 测试前确保模型文件存在
   - 图片尺寸不要过大（建议<2MB）
   - 保持图片清晰度

3. 常见问题
   - 如遇到GPU内存不足，可以减小batch_size
   - 如果识别效果不好，检查图片质量和预处理参数
   - 中文显示问题，确保系统安装了SimHei字体


## 联系方式

- 作者：缪宏雨
- 电话：19802559765
- 邮箱：m19802559765@gmail.com


## 致谢

- MNIST数据集
- PyTorch团队
- OpenCV团队
```# -
