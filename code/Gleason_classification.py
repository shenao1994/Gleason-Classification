import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from GleasonNet import GleasonNet
import matplotlib.pyplot as plt


def load_feature_data():
    dataPath = './TrainFeature-normalized image.csv'
    data = pd.read_csv(dataPath)
    row, _ = data.shape
    originData = pd.DataFrame(data)
    # print(originData)
    gleason_data = originData[originData['ClassifyValue'] != 0]
    label = gleason_data['ClassifyValue'] - 1
    label = label.reset_index()
    colNames = gleason_data[gleason_data.columns[8:]].columns
    gleason_data = gleason_data[gleason_data.columns[8:]].fillna(0)
    gleason_data = gleason_data.astype(np.float64)
    gleason_data = StandardScaler().fit_transform(gleason_data)
    gleason_data = pd.DataFrame(gleason_data)
    gleason_data.columns = colNames
    # gleason_data['label'] = label['ClassifyValue'] - 1
    # balanced Data
    smo = SMOTE(random_state=2)
    x_smote, y_smote = smo.fit_sample(gleason_data, label['ClassifyValue'])
    # print(x_smote)
    return x_smote, y_smote


def training_data(gleason_data, label):
    x_train, x_test, y_train, y_test = train_test_split(gleason_data, label, test_size=0.2, random_state=1)
    x_train = torch.from_numpy(x_train.values)
    y_train = torch.from_numpy(y_train.values)
    # x_train = Variable(x_train)
    # y_train = Variable(y_train)
    deal_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=deal_dataset, batch_size=64, shuffle=True, num_workers=2)
    glsNet = GleasonNet(input_features=321, num_class=5).double()
    # print(glsNet)
    optimizer = torch.optim.Adam(glsNet.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数设置为loss_function
    for t in range(50):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # labels = labels.unsqueeze(1)
            out = glsNet(inputs)  # 100次迭代输出
            loss = loss_func(out, labels)  # 计算loss为out和y的差异
            optimizer.zero_grad()  # 清除一下上次梯度计算的数值
            loss.backward()  # 进行反向传播
            optimizer.step()  # 最优化迭代

            if t % 2 == 0:
                plt.cla()
                print(out.argmax(dim=1))
                print(labels)
                prediction = torch.max(out, 1)[1]  ##返回每一行中最大值的那个元素，且返回其索引  torch.max()[1]， 只返回最大值的每个索引
                pred_y = prediction.data.numpy().squeeze()
                target_y = labels.data.numpy()
                plt.scatter(inputs.data.numpy()[:, 0], inputs.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
                accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
                plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    X, y = load_feature_data()
    training_data(X, y)
