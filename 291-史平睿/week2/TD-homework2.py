import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

"""
ʵ��һ�����й������ҹ��ɣ�����pytorch��ܱ�д����ѧϰģ��ѵ������
���ɣ�x��һ��5ά�����������1����������0���ڶ�����������1��
��������������2�����ĸ���������3���������������4
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # ���Բ�
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy  # loss�������ý�������ʧ

    # ��������ʵ��ǩ������lossֵ������ʵ��ǩ������Ԥ��ֵ
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    index = np.argmax(x)
    if index == 0:
        return x, 0
    elif index == 1:
        return x, 1
    elif index == 2:
        return x, 2
    elif index == 3:
        return x, 3
    else:
        return x, 4

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    #return torch.FloatTensor(X), torch.FloatTensor(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("x:", x)
    print("y:", y)
    count_dict = Counter(y.tolist())
    print("count_dict:", count_dict)
    print("����Ԥ�⼯�й��У�")
    for key, value in count_dict.items():
        print(f"{key} ������ {value} ��")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        print("y_pred:", y_pred)

        for y_p, y_t in zip(y_pred, y):
            print("y_p:", y_p)
            print("y_t:", y_t)
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("��ȷԤ�������%d, ��ȷ�ʣ�%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 20  # ѵ������
    batch_size = 20  # ÿ��ѵ����������
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)  # ����ѵ���������������Ƕ�ȡѵ����
    for epoch in range(epoch_num):  # ѵ������
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]
            loss = model.forward(x, y)  # ������ʵֵ�ͱ�ǩ������loss
            loss.backward()  # �����ݶ�
            optim.step()  # �ݶȸ���
            optim.zero_grad()  # �ݶȹ���
            watch_loss.append(loss.item())
        print("============\n��%d��ƽ��loss:%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")  # ����ģ��

    print(log)  # ��ͼ
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # ��acc����
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # ��loss����
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    #print("result:", result)
    for vec, res in zip(input_vec, result):
        print("���룺%s��Ԥ�����%d������ֵ��%f" % (vec, np.argmax(res), res[np.argmax(res)]))
        ## �洢�������������б�
        #rounded_sublist = []
        ## �������б��е�ÿ��Ԫ�ز���������
        #for i in range(len(res)):
        #    rounded_sublist.append(round(res[i]))  # �����������б�
        #print("���룺%s��Ԥ�����%s������ֵ��%s" % (vec, rounded_sublist, res))


if __name__ == "__main__":
    main()
    #test_vec, _ = build_dataset(10)
    ##print("test_vec.tolist():", test_vec.tolist())
    #predict("model.pt", test_vec)
