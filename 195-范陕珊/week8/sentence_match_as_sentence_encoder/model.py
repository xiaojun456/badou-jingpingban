# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.layer = nn.Linear(hidden_size, hidden_size)
        self.layer = BertModel.from_pretrained(config['pretrain_model_path'], local_files_only=True)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        sentence_length = torch.sum(x.gt(0), dim=-1)
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.layer(x)
        # 使用线性层
        x = self.layer(x)
        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        elif not isinstance(x, torch.Tensor):
            x = x[0]
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss()
        self.loss1=nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])

    # sentence : (batch_size, max_length)
    def forward(self, sentence_a, sentence_p=None, sentence_n=None):
        # 同时传入两个句子
        vector_a = self.sentence_encoder(sentence_a)  # vec:(batch_size, hidden_size)
        vector_p = self.sentence_encoder(sentence_p) if sentence_p is not None else None
        vector_n = self.sentence_encoder(sentence_n) if sentence_n is not None else None
        if vector_n is not None:
            # 计算loss
            # return self.loss(vector_a, vector_p, vector_n)
            return self.loss(vector_a, vector_p, vector_n)
        elif vector_p is not None:
            return self.loss1(vector_a, vector_p)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return vector_a


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())
