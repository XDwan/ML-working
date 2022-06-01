# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold


def load_dataset():
    train_dataset = pd.read_csv("data/wdbc_train.data", header=None, index_col=0)
    test_dataset = pd.read_csv("data/wdbc_test.data", header=None, index_col=0)
    validation_dataset = pd.read_csv("data/wdbc_validation.data", header=None, index_col=0)
    return train_dataset, test_dataset, validation_dataset


def dataset_split(dataset):
    label = np.array(dataset[1])
    X = np.array(dataset.iloc[:, 1:])
    return X, label


def standardization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma


def NMI(A, B):
    return metrics.normalized_mutual_info_score(A, B)


if __name__ == '__main__':
    # 导入数据集
    train_dataset, test_dataset, validation_dataset = load_dataset()
    # 分离标签和特征
    x_train, y_train = dataset_split(train_dataset)
    x_test, y_test = dataset_split(test_dataset)
    x_validation, y_validation = dataset_split(validation_dataset)

    # 标准化
    x_train_std = standardization(x_train)
    x_test_std = standardization(x_test)

    # 使用方差进行特征选择，去除方差较小的特征

    variance = np.var(x_train, axis=0)
    variance.sort()
    var_socre = []

    cluster_models = []
    for i in variance[:-7]:
        select = VarianceThreshold(threshold=i)
        x_train_sel = select.fit_transform(x_train)
        x_test_sel = select.transform(x_test)

        x_train_sel = standardization(x_train_sel)
        x_test_sel = standardization(x_test_sel)

        model = KMeans(n_clusters=2)
        model.fit(x_train_sel)
        train_pred = model.predict(x_train_sel)
        pred = model.predict(x_test_sel)
        var_socre.append(NMI(pred, y_test))
        cluster_models.append(model)

    print(np.max(var_socre))
