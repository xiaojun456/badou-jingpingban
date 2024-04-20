# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/文本分类练习.csv",
    "valid_data_path": "../data/文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\work_space\pretrain_models\models-bert-base-chinese",
    "seed": 666,
    "all_model_type":["fast_text","lstm","gru","rnn","cnn","gated_cnn","stack_gated_cnn","rcnn","bert","bert_lstm","bert_cnn","bert_mid_layer"]
}