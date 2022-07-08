# coding:utf-8
"""
@author:Finks
@time:2022/7/6 11:59

"""
import random
import pandas as pd
import paddle
import numpy as np
from sklearn.metrics import f1_score, classification_report

# load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
print(train_data.columns)
# exit()

# 去掉无用特征
drop_col_name = ['id', 'timecc',
                 'magicdmgtochamp', 'physdmgtochamp', 'truedmgtochamp',
                 'magicdmgdealt', 'physicaldmgdealt', 'truedmgdealt']
train_data = train_data.drop(drop_col_name, axis=1)
test_data = test_data.drop(drop_col_name, axis=1)

# 归一化
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
test_data = (test_data - train_data.min()) / (test_data.max() - train_data.min())
test_data = test_data.drop(['win'], axis=1)

# # shuffle
# train_data = train_data.sample(frac=1)

input_features = train_data.shape[1] - 1


# 组网
class Classifier(paddle.nn.Layer):
    def __init__(self):
        super(Classifier, self).__init__()

        # self.block1 = paddle.nn.Sequential(
        #     paddle.nn.Linear(in_features=64, out_features=64),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Linear(in_features=64, out_features=128),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Linear(in_features=128, out_features=128),
        #     paddle.nn.ReLU()
        # )
        # self.block2 = paddle.nn.Sequential(
        #     paddle.nn.Linear(in_features=128, out_features=128),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Linear(in_features=128, out_features=64),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Linear(in_features=64, out_features=32),
        #     paddle.nn.ReLU(),
        # )
        # self.block3 = paddle.nn.Sequential(
        #     paddle.nn.Linear(in_features=32, out_features=32),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Linear(in_features=32, out_features=16),
        #     paddle.nn.ReLU()
        # )

        self.fc1 = paddle.nn.Linear(in_features=input_features, out_features=64)
        self.cls = paddle.nn.Linear(in_features=64, out_features=2)
        self.relu = paddle.nn.ReLU()

        # self.att = paddle.nn.Linear(in_features=16, out_features=16)

    def forward(self, inputs):
        # print(inputs)
        x = self.relu(self.fc1(inputs))  # 64
        y = self.cls(x)

        return y


import math

from paddle.optimizer.lr import LRScheduler


class EDC(LRScheduler):
    def __init__(self,
                 learning_rate,
                 omega,
                 tau,
                 last_epoch=-1,
                 verbose=False):
        self.cycle = omega * math.pi
        self.tau = tau
        super(EDC, self).__init__(learning_rate, last_epoch, verbose)

    def edc(self, t):
        d = (math.e ** (-t / self.tau) * math.cos(self.cycle * t) + 1) / 2
        return d

    def get_lr(self):
        # print(self.last_epoch, self.last_lr)
        lr = self.base_lr * self.edc(self.last_epoch)
        return lr


# 学习率调整策略
scheduler = EDC(learning_rate=0.1, omega=0.05, tau=500, verbose=False)

model = Classifier()

model.train()
opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
loss_fn2 = paddle.nn.CrossEntropyLoss()

EPOCH_NUM = 10
BATCH_SIZE = 100

# split data
training_data = train_data.iloc[:-1000, ].values.astype(np.float32)
val_data = train_data.iloc[-1000:, ].values.astype(np.float32)


# from FGM import FGM


def out(test_data):
    """验证和输出"""
    print('测试输出'.center(32, '='))
    model.eval()
    test_data = test_data.values.astype(np.float32)
    ans = []

    for i in range(0, test_data.shape[0], 100):
        x = test_data[i:i + 100]
        features = paddle.to_tensor(x)
        predicts = model(features)
        p = paddle.argmax(predicts, axis=1).numpy()
        ans.extend(p)

    # output
    submit = pd.DataFrame({'win': ans})
    submit.to_csv('data/submission.csv', encoding='utf8', index=False)


# fgm = FGM(model)

# 训练
best_f1 = 0
for epoch_id in range(EPOCH_NUM):
    model.train()
    np.random.shuffle(training_data)

    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]

    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, 1:])

        # 归一化后向样本添加一个噪声作为增强训练
        k = random.choice([1, 2, 3])
        if k == 1:
            x = np.random.normal(x, 0.05 * x).astype(np.float32)
        elif k == 2:
            noise = np.random.normal(0, 0.05, x.shape)
            x = x + noise
            x = x.astype(np.float32)
        else:
            pass

        y = np.array(mini_batch[:, :1]).astype(np.int64)

        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)

        predicts = model(features)

        # 计算损失
        loss = loss_fn2(predicts, y, )

        avg_loss = paddle.mean(loss)

        if iter_id % 200 == 0:
            predicts = paddle.argmax(predicts, axis=1).numpy()
            target = y.flatten().astype(int).numpy()
            f1 = f1_score(target, predicts)

            print("epoch: {},"
                  " iter: {}, "
                  "loss is: {},"
                  "f1: {}".format(epoch_id, iter_id, avg_loss.numpy(), f1))

        # 反向传播
        avg_loss.backward()

        # 对抗训练
        # fgm.attack()
        predicts = model(features)

        # 计算损失
        # loss = loss_fn2(predicts, y, )
        #
        # loss_adv = paddle.mean(loss)
        # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        # fgm.restore()  # 恢复embedding参数

        opt.step()
        opt.clear_grad()
        scheduler.step()

    # 验证
    model.eval()
    val_mini_batches = [val_data[k:k + 100] for k in range(0, len(val_data), 100)]
    all_tgt, all_pred = [], []
    for iter_id, mini_batch in enumerate(val_mini_batches):
        x = np.array(mini_batch[:, 1:])
        y = np.array(mini_batch[:, :1]).astype(np.int64)

        features = paddle.to_tensor(x)
        y = paddle.to_tensor(y)

        predicts = model(features)

        predicts = paddle.argmax(predicts, axis=1).numpy()
        target = y.flatten().astype(int).numpy()
        all_tgt.extend(target)
        all_pred.extend(predicts)
    f1 = f1_score(all_tgt, all_pred)
    report = classification_report(all_tgt, all_pred)
    print(f'第{epoch_id}验证集效果：f1: {f1},\n report:\n{report}')

    # 输出结果
    if f1 > best_f1:
        best_f1 = f1
        out(test_data)
