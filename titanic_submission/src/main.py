import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# 读取
df_raw = pd.read_csv("train.csv")

# 分离特征X、标签y
y_all = df_raw["Survived"].copy()
X_all = df_raw.drop(columns=["Survived"]).copy()

# 划分训练集、测试集
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

# 数据预处理
def preprocess_data(input_df, fit_scaler = None):
    df = input_df.copy()

    # 补充空缺数据：age用中位数，embarked用众数
    age_median = df["Age"].median()
    embarked_mode = df["Embarked"].mode()[0]
    df["Age"] = df["Age"].fillna(age_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # 类别特征转换 Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # 独热编码 Embarked
    df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)
    df = df.drop(columns=["Embarked"])

    # 创建特征列表
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked_C", "Embarked_Q", "Embarked_S"]
    fea_data = df[feature_cols].copy()

    # 标准化（仅训练集，测试集维持原样）
    numeric_cols = ["Pclass", "Age", "SibSp", "Parch"]
    if fit_scaler is None:
        scaler = StandardScaler()
        fea_data[numeric_cols] = scaler.fit_transform(fea_data[numeric_cols])
        # 保存训练集得到的特征参数，传给测试集
        fit_params = {
            "age_median": age_median,
            "embarked_mode": embarked_mode,
            "scaler": scaler
        }
        return fea_data, fit_params
    else:
        scaler = fit_scaler["scaler"]
        fea_data[numeric_cols] = scaler.transform(fea_data[numeric_cols])
        return fea_data

# 训练集预处理，传给测试集
X_train_proc, preprocess_params = preprocess_data(X_train_df, fit_scaler=None)
X_test_proc = preprocess_data(X_test_df, fit_scaler=preprocess_params)

#  3.自定义PyTorch Dataset
class TitanicDataset(Dataset):
    def __init__(self, df_features, ser_label):
        self.X = torch.tensor(df_features.values, dtype=torch.float32)
        self.y = torch.tensor(ser_label.values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TitanicDataset(X_train_proc, y_train_df)
test_dataset = TitanicDataset(X_test_proc, y_test_df)

#DataLoader批量读取
BATCH_SIZE = 32   #自定单批次抽取样本数
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#  torch.nn.Module 定义二分类模型
class BinaryLogisticModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=1)   #创建线性层

    def forward(self, x): 
        #前向计算
        out = self.linear(x)
        return out

input_feature_dim = X_train_proc.shape[1]
model = BinaryLogisticModel(in_dim=input_feature_dim)

criterion = nn.BCEWithLogitsLoss()  # 二分类任务标准损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)#Adam优化器

#训练准备
EPOCHS = 600
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()          # 梯度清零
        logits = model(batch_x)        # 前向计算
        loss = criterion(logits, batch_y)
        loss.backward()              # 反向传播计算梯度
        optimizer.step()              # 参数更新

        total_train_loss += loss.item() * batch_x.size(0)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        total_train_correct += (pred == batch_y).sum().item()
        total_train_samples += batch_x.size(0)

    epoch_train_loss = total_train_loss / total_train_samples
    epoch_train_acc = total_train_correct / total_train_samples

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    total_test_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_test_loss += loss.item() * batch_x.size(0)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()
            total_test_correct += (pred == batch_y).sum().item()
            total_test_samples += batch_x.size(0)

    epoch_test_loss = total_test_loss / total_test_samples
    epoch_test_acc = total_test_correct / total_test_samples

    # 保存历史记录
    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)
    test_loss_history.append(epoch_test_loss)
    test_acc_history.append(epoch_test_acc)
    #显示关键轮次的测试结果
    if (epoch+1) % 50 == 0:
        print(f"Epoch[{epoch+1:3d}/{EPOCHS}] | Train loss:{epoch_train_loss:.4f} Acc:{epoch_train_acc:.4f} | Test loss:{epoch_test_loss:.4f} Acc:{epoch_test_acc:.4f}")

# 输出正确、错误样本 和准确率
total_test_wrong = total_test_samples - total_test_correct
final_acc = total_test_correct / total_test_samples
print("\n======== 测试集最终评估结果 ========")
print(f"测试集总样本：{total_test_samples}")
print(f"预测正确数量：{total_test_correct}")
print(f"预测错误数量：{total_test_wrong}")
print(f"测试集准确率：{final_acc:.4f}")

# 绘制loss/acc-epoch变化曲线
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(test_loss_history, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss-training epoch")
plt.legend()
plt.savefig("docs/loss_curve.png")
plt.figure()
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy-training epoch")
plt.legend()
plt.savefig("docs/acc_curve.png")
plt.show()

