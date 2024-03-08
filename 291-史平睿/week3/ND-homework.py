import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# ����pytorch�������д��ʵ��һ���������һ����nlp����
# ������ֱ�ѡȡsentence_length = 7���֣������ظ���
# ʵ��һ���˷�����������ַ�'a'�����������е�һ��λ��ʱΪ��һ�࣬
# �����ڵڶ���λ��ʱΪ�ڶ���...�����ڵ��߸�λ��ʱΪ�����࣬û������������Ϊ�ڰ��ࡣ

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # pool
        #self.pool = nn.AvgPool1d(sentence_length)
        # rnn
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length+1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)           #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)             20*7 -> 20*7*20
        #print("1:", x.shape)
        # pool
        #x = x.transpose(1,2)            #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len) 20*7*20 -> 20*20*7
        #x = self.pool(x)                #(batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, 1)       20*20*7 -> 20*20*1
        #x = x.squeeze()                 #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)                20*20*1 -> 20*20
        # rnn
        rnn_out, hidden = self.rnn(x)   #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim) 20*7*20 -> 20*7*20, 1*20*20
        #print("2:", rnn_out.shape)
        #print("3:", hidden.shape)
        #x = rnn_out[:,-1,:]             #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)          20*7*20 -> 20*20
        x = hidden.squeeze()            #(1, batch_size, vector_dim) -> (batch_size, vector_dim)                1*20*20 -> 20*20
        #print("4:", x.shape)
        y_pred = self.classify(x)       #(batch_size, vector_dim) -> (batch_size, sen_len+1)                    20*20 -> 20*8
        #print("5:", y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghij"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    # ������ֱ�ѡȡsentence_length���֣������ظ�
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #print("random choice x:", x)
    # �ַ�'a'�����ڵ�һ��λ��ʱΪ��һ�࣬�����ڵڶ���λ��Ϊ�ڶ���...
    # �����ڵ��߸�λ��Ϊ�����࣬�ַ�'a'û����Ϊ�ڰ���
    if x[0] == 'a':
        y = 0
    elif x[1] == 'a':
        y = 1
    elif x[2] == 'a':
        y = 2
    elif x[3] == 'a':
        y = 3
    elif x[4] == 'a':
        y = 4
    elif x[5] == 'a':
        y = 5
    elif x[6] == 'a':
        y = 6
    else:
        y = 7

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # ����200�����ڲ��Ե�����
    #print("evaluate x:", x, x.shape)
    #print("evaluate y:", y, y.shape)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        print("y_pred:", y_pred, y_pred.shape)
        for y_p, y_t in zip(y_pred, y):
            #print("y_p:", y_p, y_p.shape)
            #print("np.argmax(y_p):", np.argmax(y_p))
            #print("y_t:", y_t)
            if np.argmax(y_p) == y_t:# and y_t != 7:
                correct += 1
            else:
                wrong += 1
    print("��ȷԤ�������%d, ��ȷ�ʣ�%f" %(correct, correct / (correct+wrong)))
    return correct / (correct+wrong)

def main():
    epoch_num = 20         # ѵ������
    batch_size = 20        # ÿ��ѵ����������
    train_sample = 500     # ÿ��ѵ���ܹ�ѵ������������
    char_dim = 20          # ÿ���ֵ�ά��
    sentence_length = 7    # �����ı�����
    learning_rate = 0.002  # ѧϰ��

    vocab = build_vocab()  # �����ֱ�
    model = build_model(vocab, char_dim, sentence_length)             # ����ģ��
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)  # ѡ���Ż���
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # �ݶȹ���
            loss = model(x, y) # ����loss
            loss.backward()    # �����ݶ�
            optim.step()       # ����Ȩ��
            watch_loss.append(loss.item())
            #print("x:", x)
            #print("y:", y)
        print("==============\n��%d��ƽ��loss:%f"%(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    #��ͼ
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")   #��acc����
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #��loss����
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return 

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 7
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # �����ַ���
    model = build_model(vocab, char_dim, sentence_length)       # ����ģ��
    model.load_state_dict(torch.load(model_path))               # ����ѵ���õ�Ȩ��
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #���������л�
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("���룺%s, Ԥ�����%s, ����ֵ��%s" % (input_string, np.argmax(result[i]), result[i])) #��ӡ���

if __name__ == "__main__":
    main()
    test_strings = ["fcehedi", "afhegdc", "degaccd", "bddhjic"]
    predict("model.pth", "vocab.json", test_strings)
