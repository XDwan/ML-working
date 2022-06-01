# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv("data/iris_train.data", header=None)
test_data = pd.read_csv("data/iris_test.data", header=None)
validation_data = pd.read_csv("data/iris_validation.data", header=None)


def onehot(label):
    label = np.array(label)
    class_label = np.unique(label)
    onehot_code = np.zeros([label.shape[0], class_label.shape[0]])
    for idx, c in enumerate(class_label):
        onehot_code[label == c, idx] = 1
    return onehot_code


def dataset_divide(dataset):
    dataset = np.array(dataset)
    y = dataset[:, -1]
    X = dataset[:, :-1]
    return X, onehot(y)


def normalize_min(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


# 数据集划分与Min-Max归一化
train_X, train_y = dataset_divide(train_data)
train_X = normalize_min(train_X)
test_X, test_y = dataset_divide(test_data)
test_X = normalize_min(test_X)
validation_X, validation_y = dataset_divide(validation_data)
validation_X = normalize_min(validation_X)

# MLP分类器 参数（10,10） 迭代 10000次
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)
model.fit(train_X, train_y)
pred = model.predict(train_X)
print(f"train acc：{np.mean(pred == train_y)}")

pred = model.predict(test_X)
print(f"test acc：{np.mean(pred == test_y)}")

pred = model.predict(validation_X)
print(f"validation acc：{np.mean(pred == validation_y)}")
