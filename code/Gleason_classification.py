import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

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
    # print(label['ClassifyValue'].value_counts())
    # print(label)
    selectedFeatutesList = []
    # colNames = gleason_data[gleason_data.columns[8:]].columns
    for colName in gleason_data.columns[8:]:
        # if 'DWI' not in colName:
            selectedFeatutesList.append(colName)
    gleason_data = pd.DataFrame(gleason_data, columns=selectedFeatutesList)
    colNames = gleason_data.columns
    gleason_data = gleason_data.fillna(0)
    gleason_data = gleason_data.astype(np.float64)
    gleason_data = StandardScaler().fit_transform(gleason_data)
    gleason_data = pd.DataFrame(gleason_data)
    gleason_data.columns = colNames
    # smo = SMOTE(random_state=2)
    # x_smote, y_smote = smo.fit_sample(gleason_data, label['ClassifyValue'])
    # input_features = x_smote.shape[1]
    input_features = gleason_data.shape[1]
    # print(x_smote.shape[1])
    # print(x_smote)
    return gleason_data, label['ClassifyValue'], input_features


def training_data(gleason_data, label, input_num):
    cross_validation = KFold(n_splits=5, shuffle=True)
    count = 0
    # print(gleason_data)
    for train_index, test_index in cross_validation.split(np.array(gleason_data), label):
        count += 1
        log_dir = '../result/Gleason/' + 't2+adc+dwi_3d_one-hot_unbalanced_model{num}.pth'.format(num=count)
        x_train, x_test = np.array(gleason_data)[train_index], np.array(gleason_data)[test_index]
        y_train, y_test = np.array(label)[train_index], np.array(label)[test_index]
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=2)
        val_dataset = TensorDataset(x_test, y_test)
        val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=2)
        glsModel = GleasonNet(input_features=input_num, num_class=5).double().to(device)
        # print(glsNet)
        optimizer = torch.optim.Adam(glsModel.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()  # 损失函数设置为loss_function
        # loss_func = torch.nn.BCEWithLogitsLoss()
        epoch_num = 150
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
                out = glsModel(inputs)
                loss = loss_func(out, labels)  # 计算loss为out和y的差异
                # y_ordinal_encoding = transformOrdinalEncoding(labels, labels.shape[0], 5)
                # loss = loss_func(out, torch.from_numpy(y_ordinal_encoding).to(device))
                optimizer.zero_grad()  # 清除一下上次梯度计算的数值
                loss.backward()  # 进行反向传播
                optimizer.step()  # 最优化迭代
                epoch_loss += loss.item()
                print(f"{step}/{math.ceil(len(train_dataset) / train_loader.batch_size)}, train_loss: {loss.item():.4f}")
                # epoch_len = len(deal_dataset) // train_loader.batch_size
                # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
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
                        # y_pred = torch.sigmoid(y_pred)
                    # zero = torch.zeros_like(y_pred)
                    # one = torch.ones_like(y_pred)
                    # y_pred_label = torch.where(y_pred > 0.5, one, zero)
                    # y_pred_acc = (y_pred_label.sum(1)).to(torch.long)
                    kappa_value = cohen_kappa_score(y_target.to("cpu"), y_pred.argmax(dim=1).to("cpu"), weights='quadratic')
                    # kappa_value = cohen_kappa_score(y_target.to("cpu"), y_pred_acc.to("cpu"), weights='quadratic')
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
                        "current epoch: {} current Kappa: {:.4f} "
                        "current accuracy: {:.4f} best Kappa: {:.4f} at epoch {}".format(
                            epoch + 1, kappa_value, acc_metric, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()
        evaluta_model(val_dataset, log_dir, input_num)


def evaluta_model(test_data, model_name, input_n):
    device = torch.device("cpu")
    glsModel = GleasonNet(input_features=input_n, num_class=5).double().to(device)
    # Evaluate the model on test dataset #
    # print(os.path.basename(model_name).split('.')[0])
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=True, num_workers=2)
    checkpoint = torch.load(model_name)
    glsModel.load_state_dict(checkpoint['model'])
    glsModel.eval()
    print(model_name)
    with torch.no_grad():
        # saver = CSVSaver(output_dir="./output/", filename=os.path.basename(model_name).split('.')[0] + '.csv')
        for test_data in test_loader:
            test_images, test_labels = test_data
            test_images, test_labels = Variable(test_images).to(device), Variable(test_labels).to(device)
            pred = glsModel(test_images)  # Gleason Classification
            probabilities = torch.sigmoid(pred)
            # zero = torch.zeros_like(probabilities)
            # one = torch.ones_like(probabilities)
            # y_pred_ordinal = torch.where(probabilities > 0.5, one, zero)
            # y_pred_acc = (y_pred_ordinal.sum(1)).to(torch.long)
            # saver.save_batch(y_pred_acc, test_data["adcImg_meta_dict"])
            # cm = confusion_matrix(test_labels, y_pred_acc)
            cm = confusion_matrix(test_labels, probabilities.argmax(dim=1))
            # kappa_value = cohen_kappa_score(test_labels, y_pred_acc, weights='quadratic')
            kappa_value = cohen_kappa_score(test_labels, probabilities.argmax(dim=1), weights='quadratic')
            print('quadratic weighted kappa=' + str(kappa_value))
            kappa_list.append(kappa_value)
            plot_confusion_matrix(cm, model_name + '_confusion_matrix.png', title='confusion matrix')
        from sklearn.metrics import classification_report
        print(classification_report(test_labels, probabilities.argmax(dim=1), digits=4))
        accuracy_list.append(
            classification_report(test_labels, probabilities.argmax(dim=1), digits=4, output_dict=True)["accuracy"])


def transformOrdinalEncoding(labels, shape, num_class):
    ordinal_encode = np.zeros((shape, num_class))
    for i in range(0, shape):
        ordinal_encode[i:i + 1, :labels.tolist()[i]] = 1
    return ordinal_encode


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    classes = ['0', '1', '2', '3', '4']
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


if __name__ == '__main__':
    kappa_list = []
    accuracy_list = []
    X, y, input_num = load_feature_data()
    device = torch.device("cuda:0")
    training_data(X, y, input_num)
    print(accuracy_list)
    print(kappa_list)
    print('Cross_Validation Kappa:%.5f' % (sum(kappa_list) / len(kappa_list)))
    print('Cross_Validation Accuracy:%.5f' % (sum(accuracy_list) / len(accuracy_list)))
