import numpy as np
import torch
import torch.nn as nn

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)  # hidden_size��ά��һ����ڵ���input_size

    def forward(self, x):
        return self.layer(x)

class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        ht = np.zeros(self.hidden_size)
        output = []
        for xt in x:
            ux = np.dot(self.w_ih, xt)  # 5*4 4*1 -> 5*1  # 20*20 20*1 -> 20*1
            wh = np.dot(self.w_hh, ht)  # 5*5 5*1 -> 5*1  # 20*20 20*1 -> 20*1
            ht_next = np.tanh(ux+wh)
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht  # 5*6, 5*1  # 20*7, 20*1

hidden_size = 20
torch_model = TorchRNN(20, hidden_size)
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)
print("--------------------------------------------")

#x = np.array([[1,2,3,4],
#              [3,4,5,6],
#              [5,6,7,8],
#              [9,2,1,5],
#              [8,7,5,2],
#              [4,5,6,8]])
#torch_x = torch.FloatTensor(x).unsqueeze(0)
#print(torch_x.shape)
#output, hidden = torch_model.forward(torch_x)
x = np.random.randint(0, 10, size=(2,7,20))
X = torch.Tensor(x)
print(x.shape)  # (2,7,20)
print(X.shape)  # (2,7,20)
output, hidden = torch_model.forward(X)
print(output.detach().numpy(), output.shape, "torchģ��Ԥ����")         # (2,7,20)
print(hidden.detach().numpy(), hidden.shape, "torchģ��Ԥ����������")   # (1,2,20)
print("--------------------------------------------")
diy_model = DiyModel(w_ih, w_hh, hidden_size)
x = x[0]
print(x.shape)  # (7,20)
output, hidden = diy_model.forward(x)
print(output, output.shape, "diyģ��Ԥ����")        # (7,20)
print(hidden, hidden.shape, "diyģ��Ԥ����������")  # (1,20)
