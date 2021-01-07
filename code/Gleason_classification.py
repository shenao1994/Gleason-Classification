import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from code.GleasonNet import GleasonNet
import matplotlib.pyplot as plt
import math


def load_feature_data():
    dataPath = '../data/TrainFeature-normalized image.csv'
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
    x_test = torch.from_numpy(x_test.values)
    y_test = torch.from_numpy(y_test.values)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = TensorDataset(x_test, y_test)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=2)
    glsModel = GleasonNet(input_features=321, num_class=5).double().to(device)
    # print(glsNet)
    optimizer = torch.optim.Adam(glsModel.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数设置为loss_function
    epoch_num = 200
    start_epoch = 0
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(start_epoch + 1, epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        glsModel.train()
        epoch_loss = 0
        step = 0
        val_interval = 2
        for i, data in enumerate(train_loader):
            step += 1
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # labels = labels.unsqueeze(1)
            out = glsModel(inputs)  # 100次迭代输出
            loss = loss_func(out, labels)  # 计算loss为out和y的差异
            optimizer.zero_grad()  # 清除一下上次梯度计算的数值
            loss.backward()  # 进行反向传播
            optimizer.step()  # 最优化迭代
            epoch_loss += loss.item()
            print(f"{step}/{math.ceil(len(train_dataset) / train_loader.batch_size)}, train_loss: {loss.item():.4f}")
            # epoch_len = len(deal_dataset) // train_loader.batch_size
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        # scheduler.step()
        # print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            glsModel.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y_target = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data
                    val_images, val_labels = Variable(val_images).to(device), Variable(val_labels).to(device)
                    y_pred = torch.cat([y_pred, glsModel(val_images)], dim=0)
                    y_target = torch.cat([y_target, val_labels], dim=0)
                    # y_ordinal_encoding = transformOrdinalEncoding(y, y.shape[0], 5)
                    # y_pred = torch.sigmoid(y_pred)
                    # y = (y / 0.25).long()
                    print(y_target)
                kappa_value = cohen_kappa_score(y_target.to("cpu"), y_pred.argmax(dim=1).to("cpu"), weights='quadratic')
                metric_values.append(kappa_value)
                acc_value = torch.eq(y_pred.argmax(dim=1), y_target)
                # print(acc_value)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if kappa_value > best_metric:
                    best_metric = kappa_value
                    best_metric_epoch = epoch + 1
                    checkpoint = {'model': glsModel.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'epoch': epoch
                                  }
                    torch.save(checkpoint, log_dir)
                    print("saved new best metric model")
                print(
                    "current epoch: {} current Kappa: {:.4f} current accuracy: {:.4f} best Kappa: {:.4f} at epoch {}".format(
                        epoch + 1, kappa_value, acc_metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    #             plt.cla()
    #             print(out.argmax(dim=1))
    #             print(labels)
    #             prediction = torch.max(out, 1)[1]  ##返回每一行中最大值的那个元素，且返回其索引  torch.max()[1]， 只返回最大值的每个索引
    #             pred_y = prediction.data.numpy().squeeze()
    #             target_y = labels.data.numpy()
    #             plt.scatter(inputs.data.numpy()[:, 0], inputs.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
    #             accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    #             plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
    #             plt.pause(0.1)
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    log_dir = '../result/Gleason/gls_model.pth'
    X, y = load_feature_data()
    device = torch.device("cuda:0")
    training_data(X, y)
