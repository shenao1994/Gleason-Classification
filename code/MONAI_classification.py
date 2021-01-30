import sys
import logging
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, Sampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.metrics import cohen_kappa_score
import random
import math
from code.data_split import writeInCsv
from code.dataLoader import myDataset
from code.DenseNetASPP import DenseNetASPP
from torch.autograd import Variable
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
import SimpleITK as sitk

import monai
from monai.networks.nets import densenet121
from monai.transforms import Compose, LoadNiftid, AddChanneld, ScaleIntensityd, Resized, RandRotate90d, ToTensord, \
    RandFlipd, RandAffined, RandZoomd, Rotated, ResizeWithPadOrCropd, NormalizeIntensityd
from monai.metrics import compute_roc_auc
from monai.transforms import ConcatItemsd
from monai.data import CSVSaver
import types


# log_dir = 'ProstateX-2_3Modal_model4.pth'


def load_data():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    adc_path = r'E:\Work\Data\newProstate\Gleason_2d\ADC'
    t2_path = r'E:\Work\Data\newProstate\Gleason_2d\T2'
    dwi_path = r'E:\Work\Data\newProstate\Gleason_2d\DWI'
    # ProstateX_adc_train_data = r'E:\PROSTATEx-2\train\adc'
    # ProstateX_t2_train_data = r'E:\PROSTATEx-2\train\t2'
    # ProstateX_bval_train_data = r'E:\PROSTATEx-2\train\bval'
    # ProstateX_ktrans_train_data = r'E:\PROSTATEx-2\train\ktrans'
    t2_train_file_list = []
    adc_train_file_list = []
    dwi_train_file_list = []
    bval_train_file_list = []
    ktrans_train_file_list = []
    train_label_list = []
    # 2 binary labels for Gleason classification
    for t2ImgFile, adcImgFile, dwiImgFile in zip(os.listdir(t2_path),
                                                 os.listdir(adc_path),
                                                 os.listdir(dwi_path)):
        if t2ImgFile.split('.')[0][-1] != '0':
            t2_train_file_list.append(os.path.join(t2_path, t2ImgFile))
            adc_train_file_list.append(os.path.join(adc_path, adcImgFile))
            dwi_train_file_list.append(os.path.join(dwi_path, dwiImgFile))
            # ktrans_train_file_list.append(os.path.join(ProstateX_ktrans_train_data, ktransImgFile))
            train_label_list.append(int(t2ImgFile.split('.')[0][-1]) - 1)
            # train_label_list.append(0 if int(t2ImgFile.split('.')[0][-1]) < 1 else 1)
    t2_array = np.array(t2_train_file_list)
    adc_array = np.array(adc_train_file_list)
    dwi_array = np.array(dwi_train_file_list)
    train_array = np.array([t2_array, adc_array, dwi_array])
    # ktrans_array = np.array(ktrans_train_file_list)
    print(train_label_list.count(0), train_label_list.count(1), train_label_list.count(2), train_label_list.count(3),
          train_label_list.count(4))
    label_array = np.array(train_label_list)
    cross_validation = KFold(n_splits=5, shuffle=True)
    count = 0
    for train_index, test_index in cross_validation.split(train_array[0, :], label_array):
        count += 1
        t2_train, t2_test = train_array[0, :][train_index], train_array[0, :][test_index]
        adc_train, adc_test = train_array[1, :][train_index], train_array[1, :][test_index]
        dwi_train, dwi_test = train_array[2, :][train_index], train_array[2, :][test_index]
        # ktrans_train, ktrans_test = ktrans_array[train_index], ktrans_array[test_index]
        label_train, label_test = label_array[train_index], label_array[test_index]
        # print(label_train.tolist().count(0), label_train.tolist().count(1))
        train_files = [{"t2Img": t2Img, "adcImg": adcImg, "dwiImg": dwiImg, "label": label}
                       for t2Img, adcImg, dwiImg, label in
                       zip(t2_train, adc_train, dwi_train, label_train)]
        val_files = [{"t2Img": t2Img, "adcImg": adcImg, "dwiImg": dwiImg, "label": label}
                     for t2Img, adcImg, dwiImg, label in zip(t2_test, adc_test, dwi_test, label_test)]
        log_dir = '../result/Gleason/' + 't2+dwi_2d_one-hot_SPP_model{num}.pth'.format(num=count)
        # print(val_files)
        csv_name = 't2+dwi_2d_one-hot_SPP_cross_validation{num}.csv'.format(num=count)
        if not os.path.exists('../data/split_csv/' + csv_name):
            needed_val_files = val_files
            with open('../data/split_csv/' + csv_name, 'w', newline='') as f:
                writeInCsv(val_files, 'inference', f)
        else:
            csv_val_files = []
            with open('../data/split_csv/' + csv_name, 'r') as f:
                reader = csv.reader(f)
                for i in reader:
                    csv_val_files.append(eval(i[0]))
            needed_val_files = csv_val_files
            # print(needed_val_files)
        training(train_files, needed_val_files, log_dir)
        # evaluta_model(needed_val_files, log_dir)
    # DS = myDataset(t2Image_root=t2_path, adcImage_root=adc_path, dwiImage_root=dwi_path)
    # log_dir = '../result/Gleason/' + 'adc+dwi_2d_ordinal_model1.pth'
    # training(DS, log_dir)


def training(train_files, val_files, log_dir):
    # Define transforms for image
    print(log_dir)
    train_transforms = Compose(
        [
            LoadNiftid(keys=modalDataKey),
            AddChanneld(keys=modalDataKey),
            NormalizeIntensityd(keys=modalDataKey),
            # ScaleIntensityd(keys=modalDataKey),
            ResizeWithPadOrCropd(keys=modalDataKey, spatial_size=(64, 64)),
            # Resized(keys=modalDataKey, spatial_size=(48, 48), mode='bilinear'),
            ConcatItemsd(keys=modalDataKey, name="inputs"),
            RandRotate90d(keys=["inputs"], prob=0.8, spatial_axes=[0, 1]),
            RandAffined(keys=["inputs"], prob=0.8, scale_range=[0.1, 0.5]),
            RandZoomd(keys=["inputs"], prob=0.8, max_zoom=1.5, min_zoom=0.5),
            # RandFlipd(keys=["inputs"], prob=0.5, spatial_axis=1),
            ToTensord(keys=["inputs"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadNiftid(keys=modalDataKey),
            AddChanneld(keys=modalDataKey),
            NormalizeIntensityd(keys=modalDataKey),
            # ScaleIntensityd(keys=modalDataKey),
            ResizeWithPadOrCropd(keys=modalDataKey, spatial_size=(64, 64)),
            # Resized(keys=modalDataKey, spatial_size=(48, 48), mode='bilinear'),
            ConcatItemsd(keys=modalDataKey, name="inputs"),
            ToTensord(keys=["inputs"]),
        ]
    )
    # data_size = len(full_files)
    # split = data_size // 2
    # indices = list(range(data_size))
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    # full_loader = DataLoader(full_files, batch_size=64, sampler=sampler(full_files), pin_memory=True)
    # train_loader = DataLoader(full_files, batch_size=128, sampler=train_sampler, collate_fn=collate_fn)
    # val_loader = DataLoader(full_files, batch_size=split, sampler=valid_sampler, collate_fn=collate_fn)
    # DL = DataLoader(train_files, batch_size=64, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)

    # randomBatch_sizeList = [8, 16, 32, 64, 128]
    # randomLRList = [1e-4, 1e-5, 5e-5, 5e-4, 1e-3]
    # batch_size = random.choice(randomBatch_sizeList)
    # lr = random.choice(randomLRList)
    lr = 0.01
    batch_size = 256
    # print(batch_size)
    # print(lr)
    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.device)
    check_data = monai.utils.misc.first(check_loader)
    # print(check_data)
    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.device)
    # train_data = monai.utils.misc.first(train_loader)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.device)

    # Create Net, CrossEntropyLoss and Adam optimizer
    # model = monai.networks.nets.se_resnet101(spatial_dims=2, in_ch=3, num_classes=6).to(device)
    # model = densenet121(spatial_dims=2, in_channels=3, out_channels=5).to(device)
    # im_size = (2,) + tuple(train_ds[0]["inputs"].shape)
    model = DenseNetASPP(spatial_dims=2, in_channels=2, out_channels=5).to(device)
    classes = np.array([0, 1, 2, 3, 4])
    # print(check_data["label"].numpy())
    class_weights = class_weight.compute_class_weight('balanced', classes, check_data["label"].numpy())
    class_weights_tensor = torch.Tensor(class_weights).to(device)
    # print(class_weights_tensor)
    # loss_function = nn.BCEWithLogitsLoss()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    # loss_function = torch.nn.MSELoss()
    # m = torch.nn.LogSoftmax(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    # start a typical PyTorch training
    epoch_num = 300
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    # checkpoint_interval = 100
    for epoch in range(start_epoch + 1, epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        # print(scheduler.get_last_lr())
        model.train()
        epoch_loss = 0
        step = 0
        # for i, (inputs, labels, imgName) in enumerate(train_loader):
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["inputs"].to(device), batch_data["label"].to(device)
            # batch_arr = []
            # for j in range(len(inputs)):
            #     batch_arr.append(inputs[i])
            # batch_img = Variable(torch.from_numpy(np.array(batch_arr)).to(device))
            # labels = Variable(torch.from_numpy(np.array(labels)).to(device))
            # batch_img = batch_img.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            # y_ordinal_encoding = transformOrdinalEncoding(labels, labels.shape[0], 5)
            # loss = loss_function(outputs, torch.from_numpy(y_ordinal_encoding).to(device))
            loss = loss_function(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_loader) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        scheduler.step()
        print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # if (epoch + 1) % checkpoint_interval == 0:  # 每隔checkpoint_interval保存一次
        #     checkpoint = {'model': model.state_dict(),
        #                   'optimizer': optimizer.state_dict(),
        #                   'epoch': epoch
        #                   }
        #     path_checkpoint = './model/checkpoint_{}_epoch.pth'.format(epoch)
        #     torch.save(checkpoint, path_checkpoint)
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                # for i, (inputs, labels, imgName) in enumerate(val_loader):
                for val_data in val_loader:
                    val_images, val_labels = val_data["inputs"].to(device), val_data["label"].to(device)
                    # val_batch_arr = []
                    # for j in range(len(inputs)):
                    #     val_batch_arr.append(inputs[i])
                    # val_img = Variable(torch.from_numpy(np.array(val_batch_arr)).to(device))
                    # labels = Variable(torch.from_numpy(np.array(labels)).to(device))
                    # val_img = val_img.type(torch.FloatTensor).to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    # y_ordinal_encoding = transformOrdinalEncoding(y, y.shape[0], 5)
                    # y_pred = torch.sigmoid(y_pred)
                    # y = (y / 0.25).long()
                    # print(y)
                # auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                # zero = torch.zeros_like(y_pred)
                # one = torch.ones_like(y_pred)
                # y_pred_label = torch.where(y_pred > 0.5, one, zero)
                # print((y_pred_label.sum(1)).to(torch.long))
                # y_pred_acc = (y_pred_label.sum(1)).to(torch.long)
                # print(y_pred.argmax(dim=1))
                # kappa_value = kappa(cm)
                kappa_value = cohen_kappa_score(y.to("cpu"), y_pred.argmax(dim=1).to("cpu"), weights='quadratic')
                # kappa_value = cohen_kappa_score(y.to("cpu"), y_pred_acc.to("cpu"), weights='quadratic')
                metric_values.append(kappa_value)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                # print(acc_value)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if kappa_value > best_metric:
                    best_metric = kappa_value
                    best_metric_epoch = epoch + 1
                    checkpoint = {'model': model.state_dict(),
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
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation: Area under the ROC curve")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.show()
    evaluta_model(val_files, log_dir)


def evaluta_model(test_files, model_name):
    test_transforms = Compose(
        [
            LoadNiftid(keys=modalDataKey),
            AddChanneld(keys=modalDataKey),
            NormalizeIntensityd(keys=modalDataKey),
            # ScaleIntensityd(keys=modalDataKey),
            # Resized(keys=modalDataKey, spatial_size=(48, 48), mode='bilinear'),
            ResizeWithPadOrCropd(keys=modalDataKey, spatial_size=(64, 64)),
            ConcatItemsd(keys=modalDataKey, name="inputs"),
            ToTensord(keys=["inputs"]),
        ]
    )
    # create a validation data loader
    device = torch.device("cpu")
    print(len(test_files))
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=len(test_files), num_workers=2, pin_memory=torch.device)
    # model = monai.networks.nets.se_resnet101(spatial_dims=2, in_ch=3, num_classes=6).to(device)
    model = DenseNetASPP(spatial_dims=2, in_channels=2, out_channels=5).to(device)
    # Evaluate the model on test dataset #
    # print(os.path.basename(model_name).split('.')[0])
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epochs = checkpoint['epoch']
    # model.load_state_dict(torch.load(log_dir))
    model.eval()
    with torch.no_grad():
        saver = CSVSaver(output_dir="../result/GLeason/2d_output/",
                         filename=os.path.basename(model_name).split('.')[0] + '.csv')
        for test_data in test_loader:
            test_images, test_labels = test_data["inputs"].to(device), test_data["label"].to(device)
            pred = model(test_images)  # Gleason Classification
            # y_soft_label = (test_labels / 0.25).long()
            # y_soft_pred = (pred / 0.25).round().squeeze_().long()
            # print(test_data)
            probabilities = torch.sigmoid(pred)
            # pred2 = model(test_images).argmax(dim=1)
            # print(test_data)
            # saver.save_batch(probabilities.argmax(dim=1), test_data["t2Img_meta_dict"])
            # zero = torch.zeros_like(probabilities)
            # one = torch.ones_like(probabilities)
            # y_pred_ordinal = torch.where(probabilities > 0.5, one, zero)
            # y_pred_acc = (y_pred_ordinal.sum(1)).to(torch.long)
            saver.save_batch(probabilities.argmax(dim=1), test_data["dwiImg_meta_dict"])
            # print(test_labels)
            # print(probabilities[:, 1])
            # for x in np.nditer(probabilities[:, 1]):
            #     print(x)
            #     prob_list.append(x)
        # falseList = []
        # trueList = []
        # for pre, label in zip(pred2.tolist(), test_labels.tolist() ):
        #     if pre == 0 and label == 0:
        #         falseList.append(0)
        #     elif pre == 1 and label == 1:
        #         trueList.append(1)
        # specificity = (falseList.count(0) / test_labels.tolist().count(0))
        # sensitivity = (trueList.count(1) / test_labels.tolist().count(1))
        # print('specificity:' + '%.4f' % specificity + ',',
        #       'sensitivity:' + '%.4f' % sensitivity + ',',
        #       'accuracy:' + '%.4f' % ((specificity + sensitivity) / 2))
        # print(type(test_labels), type(pred))
        # fpr, tpr, thresholds = roc_curve(test_labels, probabilities[:, 1])
        # roc_auc = auc(fpr, tpr)
        # print('AUC = ' + str(roc_auc))
        # AUC_list.append(roc_auc)
        # # print(accuracy_score(test_labels, pred2))
        # accuracy_list.append(accuracy_score(test_labels, pred2))
        # plt.plot(fpr, tpr, linewidth=2, label="ROC")
        # plt.xlabel("false presitive rate")
        # plt.ylabel("true presitive rate")
        # # plt.ylim(0, 1.05)
        # # plt.xlim(0, 1.05)
        # plt.legend(loc=4)  # 图例的位置
        # plt.show()
        saver.finalize()
        # cm = confusion_matrix(test_labels, y_pred_acc)
        cm = confusion_matrix(test_labels, probabilities.argmax(dim=1))
        # cm = confusion_matrix(y_soft_label, y_soft_pred)
        # kappa_value = cohen_kappa_score(test_labels, y_pred_acc, weights='quadratic')
        kappa_value = cohen_kappa_score(test_labels, probabilities.argmax(dim=1), weights='quadratic')
        print('quadratic weighted kappa=' + str(kappa_value))
        kappa_list.append(kappa_value)
        plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    from sklearn.metrics import classification_report
    print(classification_report(test_labels, probabilities.argmax(dim=1), digits=4))
    accuracy_list.append(
        classification_report(test_labels, probabilities.argmax(dim=1), digits=4, output_dict=True)["accuracy"])
    # sensitivity = recall_score(test_labels, pred2)
    # specificity = accuracy_score(test_labels, pred2) * 2 - sensitivity
    # print(sensitivity, specificity)
    # Sensitivity_list.append(sensitivity)
    # Specificity_list.append(specificity)


def collate_fn(batch):
    return tuple(zip(*batch))


def forward_w_aspp(self, x):
    x = self.features(x)
    x = self.aspp(x)
    x = self.class_layers(x)
    return x


def soft_label(true_labels: torch.Tensor, classes: list, batch_size: int):
    error = []
    for j in range(batch_size):
        for i in range(0, 5):
            bias = -abs(true_labels[j] - classes[i])
            error.append(bias)
    error = torch.tensor(error, dtype=torch.float)
    error = error.view(batch_size, 5).cuda()
    # print(error)
    y_soft = torch.nn.functional.softmax(error, dim=1).cuda()
    return y_soft


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


if __name__ == "__main__":
    device = torch.device("cuda:0")
    modalDataKey = ["t2Img", "dwiImg"]
    kappa_list = []
    accuracy_list = []
    # AUC_list = []
    # Sensitivity_list = []
    # Specificity_list = []
    load_data()
    # print(kappa_list)
    print(accuracy_list)
    print(kappa_list)
    # print(Sensitivity_list)
    # print(Specificity_list)
    print('Cross_Validation Kappa:%.5f' % (sum(kappa_list) / len(kappa_list)))
    print('Cross_Validation Accuracy:%.5f' % (sum(accuracy_list) / len(accuracy_list)))
    # print('Cross_Validation AUC:%.5f' % (sum(AUC_list) / len(AUC_list)))
    # print('Cross_Validation Sensitivity:%.5f' % (sum(Sensitivity_list) / len(Sensitivity_list)))
    # print('Cross_Validation Specificity:%.5f' % (sum(Specificity_list) / len(Specificity_list)))
    # training(train_data, validation_data)
    # evaluta_model(validation_data)
