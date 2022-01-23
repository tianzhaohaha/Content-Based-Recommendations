import pandas as pd
import utils
import numpy as np


minhash = True
# 读取训练集
train = pd.read_csv('datasets/train_set.csv')
train.drop('timestamp', axis=1, inplace=True)

# 读取测试集
test = pd.read_csv('datasets/test_set.csv')
test.drop('timestamp', axis=1, inplace=True)
users, movies, ratings = test['userId'], test['movieId'], test['rating']


utils.recommender(train, 0, minhash, 547, 10)
# 开始预测
preds = []
for i in range(len(test)):
    print('%d/%d...' % (i+1, len(test)))
    preds.append(utils.recommender(train, 0, minhash, users[i], movies[i]))

SSE = np.sum(np.square(preds - ratings))

print(SSE)