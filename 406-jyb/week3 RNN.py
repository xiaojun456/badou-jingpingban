import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt



def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表采样sentence_length个字，有重复
    x = [np.random.choice(list(vocab.keys()),
                          p=[0.2, 0.4, 0, 0.05, 0, 0, 0.05, 0.06, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                             0, 0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.04]) for _ in
         range(sentence_length)]
    # 指定样本类别

    y = x.index('a') if 'a' in x else sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

class E_RNN(nn.Module):
    def __init__(self,char_num,sentence_length,char_dim):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=char_num,embedding_dim=char_dim)
        self.rnn=torch.nn.RNN(input_size=char_dim,hidden_size=char_dim,batch_first=True)
        self.linear=nn.Linear(in_features=char_dim,out_features=sentence_length+1)
        self.pooling=nn.AvgPool1d(kernel_size=sentence_length)
        self.softmax=nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()
    def forward(self,x,y=None):
        x=self.embedding(x)
        #x=x.transpose(1,2)
        #x=self.pooling(x)
        #x = x.squeeze()
        x,_=self.rnn(x)
        x=x[:, -1, :]
        x=self.linear(x)

        if y is not None:
            y=torch.reshape(y,[x.shape[0]])
            y=y.long()
            y_pred=x
            loss=self.loss(y_pred,y)
            return loss
        else:
            return torch.argmax(x,dim=1)

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("不同类别元素个数",torch.unique(y, return_counts=True)[1])
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(y_p)==int(y_t):
                correct+=1
            else:
                wrong += 1
                #print("真实类别:{},预测类别:{}".format(y_p, y_t))
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model =E_RNN(char_num=28,sentence_length=sentence_length,char_dim=char_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample / batch_size)):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
        # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = E_RNN(char_num=28, sentence_length=sentence_length, char_dim=char_dim)   #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "azsdfg", "rqwdeg", "nakwww","abcdgs"]
    predict("model.pth", "vocab.json", test_strings)