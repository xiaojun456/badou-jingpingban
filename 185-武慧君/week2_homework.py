# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 4)  # 线性层
        self.loss = nn.CrossEntropyLoss()  #输出采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(12)
    a = x[:3].sum()
    b = x[3:6].sum()
    c = x[6:9].sum()
    d = x[9:].sum()
    max_x = max(a, b, c, d)
    if max_x == a:
        return x, 0
    elif max_x == b:
        return x, 1
    elif max_x == c:
        return x, 2
    else:
        return x, 3


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():   # 避免计算梯度
        y_pred = model(x)  # 模型预测
        _, predicted_labels = torch.max(y_pred, 1)  # 找到预测结果中概率最高的类别
        correct = (predicted_labels == y.view(-1)).sum().item()  # 统计正确分类的样本数量
    ac = correct / test_sample_num  # 计算正确率
    print("正确预测个数：%d, 正确率：%f" % (correct, ac))
    return ac


# 测试代码

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 12  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_homework.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 12
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted_labels = torch.max(result, 1)  # 获取预测标签
        probs = F.softmax(result, dim=1)  # 将结果转换为概率值
    for vec, label, prob in zip(input_vec, predicted_labels, probs):
            class_probabilities = prob.numpy()
            print("输入：%s, 预测类别：%d, 各类别概率值：%s" % (vec, label.item(), class_probabilities.tolist()))



if __name__ == "__main__":
    main()
    test_vec = [[0.13755905, 0.35076381, 0.57084619, 0.3832996 , 0.49896093,
       0.01480904, 0.66690795, 0.77853273, 0.79793238, 0.90152417,
       0.46118564, 0.80178166],
                [0.30300801, 0.03798358, 0.81016195, 0.79105169, 0.32069,
                 0.68716979, 0.34691037, 0.66394843, 0.88578774, 0.63196328,
                 0.99597446, 0.26145962],
                [0.1658937, 0.85351591, 0.66822827, 0.08476423, 0.6636923,
                 0.80876547, 0.1670097, 0.76641777, 0.79710935, 0.69513018,
                 0.3572594, 0.61778071],
                [0.12908642, 0.70633685, 0.6015954 , 0.95443918, 0.31840862,
       0.54132937, 0.59342976, 0.3716896 , 0.68097973, 0.52497994,
       0.56560443, 0.79391497]
               ]
    predict("model_homework.pt", test_vec)

