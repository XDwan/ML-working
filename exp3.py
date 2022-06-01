# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse


def onehot(X):
    labels = np.unique(X)
    for idx, label in enumerate(labels):
        X.loc[X == label] = int(idx)
    return X.astype(int)


def standardization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


train_data = pd.read_csv("data/PRSA_train.data.csv", index_col=0)
test_data = pd.read_csv("data/PRSA_test.data.csv", index_col=0)
validation_data = pd.read_csv("data/PRSA_validation.data.csv", index_col=0)

# 去除空值
train_data = train_data.dropna()
test_data = test_data.dropna()

train_data['cbwd'] = onehot(train_data['cbwd'])
test_data['cbwd'] = onehot(test_data['cbwd'])

y_train = train_data.pop('pm2.5')
y_test = test_data.pop('pm2.5')

# 部分特征标准化
std_col = ['TEMP', 'PRES', 'Iws']
train_data.loc[:, std_col] = standardization(train_data.loc[:, std_col])
test_data.loc[:, std_col] = standardization(test_data.loc[:, std_col])

# 使用XGB回归方法
reg = XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=15, reg_lambda=4)
reg.fit(train_data, y_train)
pred = reg.predict(test_data)
print(f"MSE is {mse(y_test, pred)}")
