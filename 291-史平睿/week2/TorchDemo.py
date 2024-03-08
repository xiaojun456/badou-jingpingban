import torch 
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.linear(x)
        print(x)
        y_pred = self.activation(x)
        print(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# ����һ������, ���������ɷ���������������Ҫѧϰ�Ĺ���
# �������һ��5ά�����������һ��ֵ���ڵ����ֵ����Ϊ������������֮Ϊ������
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

# �������һ������
# ����������������
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("����Ԥ�⼯�й���%d����������%d��������" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):
		    #print("y_p:", y_p)
            #print("y_t:", y_t)
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            if float(y_p) > 0.5 and int(y_t) == 1:
                correct += 1 
            else:
                wrong += 1
        print("��ȷԤ�������%d�� ��ȷ�ʣ�%f" % (correct, correct/(correct+wrong)))
        return correct/(correct+wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log=[]
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("===========\n��%d��ƽ��loss:%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
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
    for vec, res in zip(input_vec, result):
        print("���룺%s�� Ԥ�����%d�� ����ֵ��%f" % (vec, round(float(res)), res))

if __name__ == "__main__":
    #main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
            [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
            [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
            [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)