import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import time
from torch.autograd import Variable
from code.dataLoader import myDataset
import torch.utils.data as Data
import glob
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import SimpleITK as sitk


def loadData():
    T2Path = r'E:\data\No1HospitalProstate\Gleason_label2d\T2'
    ADCPath = r'E:\data\No1HospitalProstate\Gleason_label2d\ADC'
    DWIPath = r'E:\data\No1HospitalProstate\Gleason_label2d\DWI'
    # Get all the names of the training data
    # T2_files = glob.glob(os.path.join(T2Path, '*.nii'))
    # ADC_files = glob.glob(os.path.join(ADCPath, '*.nii'))
    # DWI_files = glob.glob(os.path.join(DWIPath, '*.nii'))
    # train_files = [T2_files, ADC_files, DWI_files]
    DS = myDataset(t2Image_root=T2Path, adcImage_root=ADCPath, dwiImage_root=DWIPath)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    return DL


class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor=True):
        super(MyDenseNetConv, self).__init__()
        in_channel = 3
        original_model = torchvision.models.densenet201(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        self.features[0] = torch.nn.Conv2d(in_channel, 32, kernel_size=7, stride=2, padding=2)
        print(self.features[0])
        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x


class MyDenseNetDens(torch.nn.Module):
    def __init__(self, nb_out=32):
        super().__init__()
        self.dens1 = torch.nn.Linear(in_features=32, out_features=6144)
        self.dens2 = torch.nn.Linear(in_features=6144, out_features=256)
        self.dens3 = torch.nn.Linear(in_features=256, out_features=nb_out)

    def forward(self, x):
        x = self.dens1(x)
        x = torch.nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens2(x)
        x = torch.nn.functional.selu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.dens3(x)
        return x


class MyDenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mrnc = MyDenseNetConv()
        self.mrnd = MyDenseNetDens()

    def forward(self, x):
        x = self.mrnc(x)
        x = self.mrnd(x)
        return x


def predict(dset_loaders, model, use_gpu=False):
    predictions = []
    labels_lst = []

    # ii_n = len(dset_loaders)
    start_time = time.time()
    for i, (inputs, labels, imgName) in enumerate(dset_loaders):
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs.float())
        labels = Variable(labels)
        outputs = model(inputs).data
        print(outputs)
        for j in range(outputs.size(0)):
            print(imgName[j])
            img = sitk.GetImageFromArray(outputs[j].numpy())
            sitk.WriteImage(img, './outputImg/' + imgName[j].split('\\')[-1])
        # clf = svm.SVC(C=0.9, kernel='rbf')
        # clf.fit(model(inputs).data, labels)
        # accuracy_svm = clf.score(model(inputs).data, labels)
        # print(accuracy_svm)
        # print(classification_report(labels, clf.predict(model(inputs).data), digits=4))
        predictions.append(outputs)
        labels_lst.append(labels)
        # print('\rpredict: {}/{}'.format(i, ii_n - 1), end='')
    print('Execution time {0:.2f} s'.format(round(time.time() - start_time), 2))
    # if len(predictions) > 0:
    # return {'pred': torch.cat(predictions, 0), 'true': torch.cat(labels_lst, 0)}


def save_prediction(path2data, featureOutput):
    for key in featureOutput.keys():
        if featureOutput[key][0].is_cuda:
            data = {'true': featureOutput[key][0].cpu().numpy(),
                    'pred': featureOutput[key][1].cpu().numpy()}
        else:
            data = {'true': featureOutput[key][0].numpy(),
                    'pred': featureOutput[key][1].numpy()}
        if not os.path.exists(path2data + key):
            os.makedirs(path2data + key)

        print('\nSaving ' + featureOutput[key][2] + ' ' + key)
        np.savez(path2data + key + "/" + featureOutput[key][2] + ".npz", **data)
        print('Saved in:' + path2data + key + "/" + featureOutput[key][2] + ".npz")


if __name__ == '__main__':
    use_gpu = torch.cuda.device_count() > 0
    model = MyDenseNet()
    if use_gpu:
        print("Using all GPU's ")
        model = torch.nn.DataParallel(model)
        model.cuda()
        convnet = model.module.mrnc
    else:
        # model = torch.nn.DataParallel(model)
        convnet = model.mrnc
        print("Using CPU's")
    predict(loadData(), convnet, use_gpu=use_gpu)
    # print(convOutput_valid['true'].size(), convOutput_valid['pred'].size())
    # model_name = 'MyDenseNet'
    # sav_feats = {
    #     # 'train': (convOutput_train['pred'], convOutput_train['true'], model_name),
    #     'valid': (convOutput_valid['pred'], convOutput_valid['true'], model_name),
    #     # 'test': (convOutput_test['pred'], convOutput_test['true'], model_name)
    # }
    # print(convOutput_valid)
    # save_prediction('./results/', sav_feats)
