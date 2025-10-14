
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    # --- 1. 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # --- 2. 数据预处理与加载 (适配EfficientNet) ---
    # EfficientNet预训练模型使用的均值和标准差
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # EfficientNet-B0 的标准输入尺寸是 224x224
    input_size = 224

    # 训练集的数据增强和标准化
    transform_train = transforms.Compose([
        transforms.Resize(input_size), # 将32x32的图像上采样到224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # 测试集也需要进行同样的尺寸调整和标准化
    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # 下载并加载数据集
    data_path = './data'
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    # 创建 DataLoader
    batch_size = 64 # EfficientNet更大，适当减小batch size以防显存不足
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    print(f"数据集类别: {class_names}")

    # --- 3. 模型定义与修改 (使用EfficientNet-B0) ---
    # 加载在ImageNet上预训练的EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # 冻结所有预训练层的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层 (分类头)
    # EfficientNet的分类器是一个叫做 'classifier' 的Sequential层
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    # --- 4. 优化器和损失函数的定义 ---
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 只优化我们新添加的分类头的参数
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # 定义一个学习率调度器，在15个epoch后降低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


    # --- 5. 训练与评估函数 (保持不变) ---
    def evaluate_model(model_to_eval, dataloader):
        model_to_eval.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model_to_eval(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    # --- 6. 训练主循环 ---
    num_epochs = 20
    start_time = time.time()

    print("\n开始使用 EfficientNet-B0 进行训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        
        epoch_loss = running_loss / len(train_dataset)
        test_accuracy = evaluate_model(model, test_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {epoch_loss:.4f} | "
              f"Test Accuracy: {test_accuracy:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n训练完成！总耗时: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")

    # --- 7. 最终结果输出 ---
    final_accuracy = evaluate_model(model, test_loader)
    print("\n" + "="*50)
    print(f"最终在CIFAR-10测试集上的准确率: {final_accuracy:.2f}%")
    print("="*50)


if __name__ == '__main__':
    main()
