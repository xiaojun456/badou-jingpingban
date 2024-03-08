# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""

#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 128)  # 线性层
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 5)
        self.activate = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        print("TorchModel模型，输入参数x")
        print(x)
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        x = self.activate(x)
        x = self.linear1(x)
        x = self.linear2(x)
        y_pred = x
        print("TorchModel模型，输入参数y_pred # (batch_size, 1) -> (batch_size, 1)")
        print(y_pred)
        if y is not None:
            print("TorchModel模型，输入参数真实标签y")
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果



# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，第一个最大，就属于第一类，第二个最大就属于第二类，第三个最大就属于第三类，第四个最大就属于第四类，第五个最大就属于第五类
def build_sample():
    # 生成1到5的随机序列
    vector = np.random.choice(np.arange(1, 6), size=5, replace=False)
    # 获取最大数值的索引编号
    max_index = np.argmax(vector)
    # 将最大数值的索引位置置为1，其他位置置为0
    classified_vector = np.zeros_like(vector)
    classified_vector[max_index] = 1
    return vector, classified_vector, max_index



# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    vector = []
    classified_vector = []
    max_index = []
    for i in range(total_sample_num):
        x, y, z = build_sample()
        vector.append(x)
        classified_vector.append(y)
        max_index.append(z)
    return torch.FloatTensor(vector), torch.FloatTensor(classified_vector), torch.FloatTensor(max_index)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    test_data, test_labels_vector, test_labels = build_dataset(test_sample_num)
    # 使用嵌套的列表推导式来找到包含1的元素的索引
    indices = test_labels
    # 输出结果
    print(f"包含1的元素的索引位置: {indices}")
    from collections import Counter
    # 假设你有一个5分类的列表
    labels = indices
    # 使用Counter统计每个分类出现的次数
    label_counts = Counter(labels)
    # 计算总数
    total = sum(label_counts.values())
    # 打印每个分类出现的次数
    for label, count in label_counts.items():
        percentage = count / total * 100
        print(f"分类 {label} 出现了 {count} 次，占比 {percentage:.2f}%")
    with torch.no_grad():
        outputs = model(test_data)
        print("预测结果outputs")
        print(outputs)
        yyy, predicted = torch.max(outputs.data, 1)
        print("预测结果 _, predicted = torch.max(outputs.data, 1)")
        print(yyy)
        print(predicted)
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

    print(f'Test Accuracy: {accuracy}')
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y, _ = build_dataset(train_sample)
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
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        outputs = model(input_vec)  # 模型预测
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

    print(f'predict Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
    test_data, test_labels_vector, test_labels = build_dataset(500)
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_data)
