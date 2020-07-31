import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from gensim.models import Word2Vec

path = "./Data/"
def load_training_data(path):
    # 读取 training 需要的数据
    # 如果是 'training_label.txt'，需要读取 label，如果是 'training_nolabel.txt'，不需要读取 label
    if 'training_label' in path:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            # lines是二维数组，第一维是行line(按回车分割)，第二维是每行的单词(按空格分割)
            lines = [line.strip('\n').split(' ') for line in lines]
        # 每行按空格分割后，第2个符号之后都是句子的单词
        x = [line[2:] for line in lines]
        # 每行按空格分割后，第0个符号是label
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            # lines是二维数组，第一维是行line(按回车分割)，第二维是每行的单词(按空格分割)
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path):
    # 读取 testing 需要的数据
    with open(path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        # 第0行是表头，从第1行开始是数据
        # 第0列是id，第1列是文本，按逗号分割，需要逗号之后的文本
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    # outputs => 预测值，概率（float）
    # labels => 真实值，标签（0或1）
    outputs[outputs>=0.5] = 1 # 大于等于 0.5 为正面
    outputs[outputs<0.5] = 0 # 小于 0.5 为负面
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


def train_word2vec(x):
    # 训练 word to vector 的 word embedding
    # window：滑动窗口的大小，min_count：过滤掉语料中出现频率小于min_count的词
    model = Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

# 读取 training 数据
print("loading training data ...")
path = './Data/'
train_x, y = load_training_data(path + '/training_label.txt')
train_x_no_label = load_training_data(path + '/training_nolabel.txt')

# 读取 testing 数据
print("loading testing data ...")
test_x = load_testing_data(path + '/testing_data.txt')
print("loading data end")

# 把 training 中的 word 变成 vector
model = train_word2vec(train_x + train_x_no_label + test_x) # w2v_all

print("saving model ...")
model.save(path + '/w2v_all.model')
print("save model end")

