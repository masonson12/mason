# 自动化与电气工程学院
# 彭佳俊
# 12241626
# 大作业：针对 MNIST 数据集进行图像分类任务

# 方法1
# 使用 PyTorch 框架，主要针对 MNIST 数据集进行图像分类任务。
# 使用卷积神经网络（CNN）进行 MNIST 手写数字的分类任务。
# 涵盖了从数据加载、模型定义、训练与验证、测试评估、混淆矩阵、ROC 曲线等一系列步骤。

# 首先，通过数据增强（如随机旋转和水平翻转）来提高模型的鲁棒性，避免过拟合。
# 然后，定义了一个简单的卷积神经网络架构，包括卷积层、池化层和全连接层，并在训练过程中使用L2正则化以进一步减少过拟合风险。
# 在训练过程中，除了损失和准确率，还计算了F1分数、精确度（Precision）和召回率（Recall）等常见评价指标，以全面评估模型的性能。
# 同时，设计了验证集的评估机制，避免模型仅在训练集上表现良好，确保其泛化能力。
# 通过混淆矩阵、ROC曲线和分类报告的绘制，进一步深入分析模型在不同类别上的表现，帮助识别可能存在的偏差。
# 此外，为了提高模型评估的直观性，代码包含了多个可视化功能，如损失曲线、F1分数曲线、精确度曲线和召回率曲线，便于在训练过程中实时监控模型的表现。
# 最终，模型经过训练和评估后，会保存最佳权重，并在测试集上进行详细评估，生成分类报告并可选择保存为Excel格式。

# *** 若没有cuda可能训练较慢，推荐使用cuda***
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
import pandas as pd

# 检查 GPU 可用性
# 使用GPU（如果可用）或者CPU，
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'*****方法1****')
print(f'使用设备: {device}')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义超参数
BATCH_SIZE = 64  # 每批次数据量
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 10  # 训练轮数
L2_REGULARIZATION = 0.001  # L2 正则化系数

# 数据增强和归一化处理
transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转图像
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 按比例分割训练集和验证集
val_size = int(0.1 * len(train_dataset))  # 验证集大小为10%
train_size = len(train_dataset) - val_size  # 剩余部分为训练集
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# DataLoader，批量加载数据
train_loader = DataLoader(dataset=train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义简单的卷积神经网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 第一个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层
        self.fc3 = nn.Linear(64, 10)  # 输出层
        self.dropout = nn.Dropout(0.5)  # Dropout层，用于防止过拟合
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 卷积 -> 激活 -> 池化
        x = x.view(-1, 32 * 14 * 14)  # 展平数据
        x = torch.relu(self.fc1(x))  # 第一层全连接 -> ReLU激活
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc2(x))  # 第二层全连接 -> ReLU激活
        x = self.fc3(x)  # 输出层，返回分类结果
        return x

# 定义训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("训练开始...")
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_precision_scores, val_precision_scores = [], []
    train_recall_scores, val_recall_scores = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_train_labels, all_train_predictions = [], []
        # 训练过程中，遍历训练数据加载器
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + L2_REGULARIZATION * sum(p.pow(2).sum() for p in model.parameters())  # L2 正则化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
        train_f1_scores.append(train_f1)
        train_precision = precision_score(all_train_labels, all_train_predictions, average='weighted')
        train_precision_scores.append(train_precision)
        train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted')
        train_recall_scores.append(train_recall)

        # 验证模型
        model.eval()  # 设置模型为验证模式
        val_loss, correct, total = 0, 0, 0
        all_val_labels, all_val_predictions = [], []
        
        with torch.no_grad(): # 禁用梯度计算
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
        # 计算并保存验证集的平均损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        # 计算 F1 分数、精确度和召回率
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
        val_f1_scores.append(val_f1)
        val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
        val_precision_scores.append(val_precision)
        val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')
        val_recall_scores.append(val_recall)
        # 输出每轮的损失和准确率
        print(f'第 {epoch + 1} 轮 | 训练损失: {avg_loss:.4f}, 训练准确率: {accuracy:.2f}% | '
              f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies, train_f1_scores, val_f1_scores, \
           train_precision_scores, val_precision_scores, train_recall_scores, val_recall_scores

# 实例化模型并转移到GPU
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练和验证模型
train_losses, train_accuracies, val_losses, val_accuracies, train_f1_scores, val_f1_scores, \
train_precision_scores, val_precision_scores, train_recall_scores, val_recall_scores = train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

# 保存模型
torch.save(model.state_dict(), 'simple_cnn.pth')

# 测试模型并打印分类报告
def test_model(model, test_loader, criterion):
    model.eval()
    all_labels, all_predictions = [], []
    correct, total, test_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%')

    # 打印详细分类报告
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)], output_dict=True)
    print(report)

    return all_labels, all_predictions

all_labels, all_predictions = test_model(model, test_loader, criterion)

# 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_predictions):
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.title('混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.show()

plot_confusion_matrix(all_labels, all_predictions)

# 绘制损失曲线
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, color='blue', label='训练损失')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, color='orange', label='验证损失')
    plt.xlabel('轮次')
# 绘制F1分数曲线
def plot_f1_score(train_f1_scores, val_f1_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_f1_scores, color='purple', label='训练F1分数')
    plt.plot(range(1, NUM_EPOCHS + 1), val_f1_scores, color='brown', label='验证F1分数')
    plt.xlabel('轮次')
    plt.ylabel('F1分数')
    plt.title('训练和验证F1分数曲线')
    plt.legend()
    plt.show()

plot_f1_score(train_f1_scores, val_f1_scores)

# 绘制精确度（Precision）曲线
def plot_precision(train_precision_scores, val_precision_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_precision_scores, color='cyan', label='训练精确度')
    plt.plot(range(1, NUM_EPOCHS + 1), val_precision_scores, color='magenta', label='验证精确度')
    plt.xlabel('轮次')
    plt.ylabel('精确度')
    plt.title('训练和验证精确度曲线')
    plt.legend()
    plt.show()

plot_precision(train_precision_scores, val_precision_scores)

# 绘制召回率（Recall）曲线
def plot_recall(train_recall_scores, val_recall_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_recall_scores, color='orange', label='训练召回率')
    plt.plot(range(1, NUM_EPOCHS + 1), val_recall_scores, color='green', label='验证召回率')
    plt.xlabel('轮次')
    plt.ylabel('召回率')
    plt.title('训练和验证召回率曲线')
    plt.legend()
    plt.show()

plot_recall(train_recall_scores, val_recall_scores)

# 计算并绘制ROC曲线
from sklearn.metrics import roc_auc_score

def plot_roc_curve(all_labels, all_predictions, num_classes=10):
    # 计算每个类别的ROC AUC
    roc_auc = dict()
    fpr = dict()
    tpr = dict()
    for i in range(num_classes):
        # 为每个类计算 ROC 曲线
        fpr[i], tpr[i], _ = roc_curve([1 if label == i else 0 for label in all_labels],
                                       [1 if pred == i else 0 for pred in all_predictions])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='类 {} (AUC = {:.2f})'.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.50)')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正性率 (TPR)')
    plt.title('多类ROC曲线')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(all_labels, all_predictions)

# 绘制测试集的分类报告（可以选择是否保存）
def plot_classification_report(all_labels, all_predictions):
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)], output_dict=True)
    
    # 将分类报告转换为DataFrame并显示
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # 如果需要，可以将分类报告保存为Excel
    report_df.to_excel('classification_report.xlsx')

plot_classification_report(all_labels, all_predictions)

# 测试并生成最终报告
def final_test_and_report(model, test_loader, criterion):
    print("\n开始测试集评估...")
    model.eval()
    all_labels, all_predictions = [], []
    correct, total, test_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%')

    # 打印详细分类报告
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)], output_dict=True)
    print(report)

    # 保存测试结果
    return all_labels, all_predictions

# 调用测试函数并生成最终报告
final_test_and_report(model, test_loader, criterion)

# 显示预测结果和真实图像
def show_predictions(test_loader, model, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f'真实: {labels[i].item()}\n预测: {predicted[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_predictions(test_loader, model)


# 方法2 使用ResNet18模型

# 基于PyTorch的深度学习模型，用于训练和评估MNIST数据集上的数字分类任务
# 1.数据预处理与增强：为了适配预训练的ResNet18模型，首先对MNIST数据集进行灰度转RGB处理，确保输入数据有三个通道。接着，通过常见的预处理操作（如标准化）来处理数据，以便与预训练模型的输入要求一致。

# 2.模型架构：使用了预训练的ResNet18模型，并通过修改其最后一层全连接层将输出类别数调整为10（MNIST的类别数）。通过迁移学习的方式利用预训练权重，提升模型性能。

# 3.训练过程：采用带动量的SGD优化器，并使用学习率调度器（StepLR）进行动态调整，逐步降低学习率，从而更好地优化模型。训练过程中计算并记录损失、准确率、精确率、召回率和F1分数，方便评估模型性能。

# 4.模型评估：每个epoch结束后，会在测试集上评估模型，并计算多项指标（准确率、精确率、召回率和F1分数），这些指标不仅帮助监控训练进展，还为最终的模型选择提供依据。

# 5.可视化与分析：通过绘制训练损失、准确率、精确率、召回率和F1分数随epoch变化的曲线，帮助直观地分析模型的训练情况。此外，还展示了最终的混淆矩阵和部分预测图像，进一步分析模型的分类效果。

# 6.模型保存：最后，将训练完成的模型参数保存到文件中，以便后续加载和使用。

# *** 若没有cuda可能训练较慢，推荐使用cuda***
# ResNet（残差网络）由于其深层次的网络结构和跳跃连接（shortcut connections），
# 训练时会比简单的卷积神经网络（CNN）耗费更多的时间和计算资源。
# ResNet训练会比较慢一点，故此所花时间更多，但是能够显著提高分类的准确性
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道（以适应ResNet）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的 ResNet18 模型
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 或者使用 DEFAULT

# 修改最后的全连接层以适应你的分类任务（MNIST是10个类别）
model.fc = nn.Linear(model.fc.in_features, 10)

# 将模型放到 GPU 上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f'*****方法2****')
print(f'使用设备: {device}')

# 使用带动量的 SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 使用学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练过程
num_epochs = 5
train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1_scores = []

test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []

# 用于存储测试集的预测图像，仅在最后一个 epoch 显示
test_images = []
test_labels = []
test_preds = []

# 训练过程
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 记录所有的预测和标签
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    scheduler.step()  # 更新学习率

    # 计算训练集的精确率、召回率和F1-score
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    train_precisions.append(precision)
    train_recalls.append(recall)
    train_f1_scores.append(f1)

    # 打印每个epoch的信息
    print(f'开始训练......')
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_accuracy:.2f}%, "
          f"Precision: {precision:.2f}, "
          f"Recall: {recall:.2f}, "
          f"F1 Score: {f1:.2f}, "
          f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # 在测试集上评估模型
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录所有的预测和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 仅在最后一个epoch收集前10张图像用于显示
            if len(test_images) < 10:  # 限制显示的图像数量
                test_images.extend(inputs.cpu().numpy()[:10 - len(test_images)])
                test_labels.extend(labels.cpu().numpy()[:10 - len(test_labels)])
                test_preds.extend(predicted.cpu().numpy()[:10 - len(test_preds)])

    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)

    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    test_f1_scores.append(test_f1)

    print(f"Test Accuracy: {test_accuracy:.2f}%, "
          f"Test Precision: {test_precision:.2f}, "
          f"Test Recall: {test_recall:.2f}, "
          f"Test F1 Score: {test_f1:.2f}")

    # 清理显存
    torch.cuda.empty_cache()

# 绘制训练损失、准确率、精确率、召回率和F1分数曲线
plt.figure(figsize=(12, 8))

# 训练损失曲线绘制
plt.subplot(2, 3, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Epoch')
plt.grid(True)

# 准确率曲线绘制
plt.subplot(2, 3, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train and Test Accuracy vs Epoch')
plt.legend()
plt.grid(True)

# 精确率曲线绘制
plt.subplot(2, 3, 3)
plt.plot(range(1, num_epochs+1), train_precisions, label='Train Precision', color='blue')
plt.plot(range(1, num_epochs+1), test_precisions, label='Test Precision', color='red')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision vs Epoch')
plt.legend()
plt.grid(True)

# 召回率曲线绘制
plt.subplot(2, 3, 4)
plt.plot(range(1, num_epochs+1), train_recalls, label='Train Recall', color='green')
plt.plot(range(1, num_epochs+1), test_recalls, label='Test Recall', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall vs Epoch')
plt.legend()
plt.grid(True)

# F1分数曲线绘制
plt.subplot(2, 3, 5)
plt.plot(range(1, num_epochs+1), train_f1_scores, label='Train F1 Score', color='orange')
plt.plot(range(1, num_epochs+1), test_f1_scores, label='Test F1 Score', color='purple')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()

# 在最后一个epoch后显示 10 张预测图
if num_epochs > 0:
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_images[i].transpose(1, 2, 0))  # 转换为 HWC 格式
        ax.set_title(f'Pred: {test_preds[i]} | True: {test_labels[i]}')
        ax.axis('off')
    plt.show()

# 保存训练好的模型
torch.save(model.state_dict(), 'resnet18_mnist.pth')
