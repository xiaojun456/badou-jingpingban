# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_text_cat.csv",
    "valid_data_path": "data/val_text_cat.csv",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path": r"D:\bert-base-chinese",
    "seed": 987
}
