from radiomics import featureextractor
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, RepeatedKFold, validation_curve, GridSearchCV, StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV, RFE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LinearRegression

from scipy.stats import levene, ttest_ind
from imblearn.over_sampling import SMOTE
import SimpleITK as sitk
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle
# import pymrmr
# import mifs


flip_up_down = iaa.Sequential([
    iaa.Flipud(1)
])


def get_features():
    false_dataPath = r'E:\data\No2HospitalProstate\buliangbingli\binglishengjiF'
    positive_dataPath = r'E:\data\No2HospitalProstate\buliangbingli\binglishengjiP'
    featureExcelPath = './No2Hospital_binglishengji_features.csv'
    paramPath = './Params.yaml'
    yaml_params = {'binWidth': 20, 'sigma': [1, 2, 3], 'verbose': True}
    # Instantiate the extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**yaml_params)
    # extractor.enableInputImageByName('LoG')
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeaturesByName(glcm=['Autocorrelation', 'Homogeneity1', 'SumSquares'])
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    false_feature_values, t2_feature_names, adc_feature_names = loop_different_folds(false_dataPath, '0', extractor)
    positive_feature_values, _, _ = loop_different_folds(positive_dataPath, '1', extractor)
    total_feature_values = false_feature_values + positive_feature_values
    total_feature_keys = ['CaseName', 'label'] + t2_feature_names + adc_feature_names
    df = pd.DataFrame(total_feature_values, columns=total_feature_keys)
    df.to_csv(featureExcelPath, index=False)


def loop_different_folds(dataPath, label, extractor):
    feature_values_list = []
    for foldName in os.listdir(dataPath):
        t2_imgPath = os.path.join(dataPath, foldName + '\\T2_img.nii')
        t2_labelPath = os.path.join(dataPath, foldName + '\\T2_Merge.nii')
        adc_imgPath = os.path.join(dataPath, foldName + '\\ADC_img.nii')
        adc_labelPath = os.path.join(dataPath, foldName + '\\ADC_Merge.nii')
        t2_feature_keys, t2_feature_values = arrangeFeatures(extractor, t2_imgPath, t2_labelPath, 'T2')
        adc_feature_keys, adc_feature_values = arrangeFeatures(extractor, adc_imgPath, adc_labelPath, 'ADC')
        feature_values = [foldName, label] + t2_feature_values + adc_feature_values
        feature_values_list.append(feature_values)
    return feature_values_list, t2_feature_keys, adc_feature_keys


def arrangeFeatures(extractor, imgPath, labelPath, key_name):
    print(imgPath)
    featureVector = extractor.execute(imgPath, labelPath)
    feature_names = []
    feature_values = []
    for feature_name, feature_value in zip(list(featureVector.keys()), list(featureVector.values())):
        if 'diagnostics' in feature_name:
            continue
        feature_names.append(key_name + '_' + feature_name)
        feature_values.append(feature_value)
    return feature_names, feature_values


def getNiiImg():
    dataPath = r'E:\data\No2HospitalProstate\buliangbingli\binglishengjiP'
    for foldName in os.listdir(dataPath):
        print(foldName)
        adc_reader = sitk.ImageSeriesReader()
        adc_series = adc_reader.GetGDCMSeriesFileNames(dataPath + '\\' + foldName + '\\ADC')
        adc_reader.SetFileNames(adc_series)
        adcImage = adc_reader.Execute()
        t2_reader = sitk.ImageSeriesReader()
        t2_series = t2_reader.GetGDCMSeriesFileNames(dataPath + '\\' + foldName + '\\T2')
        t2_reader.SetFileNames(t2_series)
        t2Image = t2_reader.Execute()
        sitk.WriteImage(adcImage, dataPath + '\\' + foldName + '\\ADC_img.nii')
        sitk.WriteImage(t2Image, dataPath + '\\' + foldName + '\\T2_img.nii')


def fix_data_axis():
    dataPath = r'E:\data\finalData'
    saveFold = r'E:\data\No1HospitalProstate\fix_final_data'
    for foldName in os.listdir(dataPath):
        imgPath = os.path.join(dataPath, foldName + '\\DWI_src_tumor.nii.gz')
        # labelPath = os.path.join(dataPath, foldName + '\\T2_label_tumor.nii.gz')
        img = sitk.ReadImage(imgPath)
        # label = sitk.ReadImage(labelPath)
        print(img.GetDirection())
        img_fixDirection = img.GetDirection()[:4] + (-img.GetDirection()[4],) + img.GetDirection()[5:]
        print(img_fixDirection)
        img.SetDirection(img_fixDirection)
        # label.SetDirection(img_fixDirection)
        imgData = sitk.GetArrayFromImage(img)
        # labelData = sitk.GetArrayFromImage(label)
        filpup_batch_img = flip_up_down(images=imgData)
        # filpup_batch_label = flip_up_down(images=labelData)
        outImg = sitk.GetImageFromArray(filpup_batch_img)
        # outLabel = sitk.GetImageFromArray(filpup_batch_label)
        saveFoldPath = os.path.join(saveFold, foldName)
        if not os.path.exists(saveFoldPath):
            os.mkdir(saveFoldPath)
        else:
            pass
        ImgSavePath = os.path.join(saveFoldPath, 'DWI_Img.nii')
        # LabelSavePath = os.path.join(saveFoldPath, 'ROI.nii')
        # print(ImgSavePath)
        sitk.WriteImage(outImg, ImgSavePath)
        # sitk.WriteImage(outLabel, LabelSavePath)


def load_data():
    dataPath = '../data/No2Hospital_binglishengji_features.csv'
    data = pd.read_csv(dataPath, encoding='gbk')
    row, _ = data.shape
    originData = pd.DataFrame(data)
    # classType = 3
    # originData['label'] = originData.apply(lambda x: 0 if x.ClassifyValue < classType else 1, axis=1)
    # data1 = originData[originData['ClassifyValue'] < classType]
    # data2 = originData[originData['ClassifyValue'] >= classType]
    data1 = originData[originData['病理升级'] == 0]
    data2 = originData[originData['病理升级'] == 1]
    return data1, data2, originData


class ClassificationProcessing:
    @staticmethod
    def TTest_Lasso(data):
        selectedFeatutesList = []
        label = data['病理升级']
        print(label[label[0:] == 0].values.size, label[label[0:] == 1].values.size)
        colNames = data[data.columns[9:]].columns
        data = data[data.columns[9:]].fillna(0)
        data = data.astype(np.float64)
        data = StandardScaler().fit_transform(data)
        data = pd.DataFrame(data)
        data.columns = colNames
        data['label'] = label
        # balanced Data
        smo = SMOTE(random_state=2)
        X_smote, y_smote = smo.fit_sample(data, data['label'])
        for colName in X_smote.columns[0:-1]:
            # if 'ADC' not in colName:
                if levene(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[1] > 0.05 \
                    and ttest_ind(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[1] < 0.05:selectedFeatutesList.append(colName)
                elif levene(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[1] <= 0.05 and \
                        ttest_ind(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName],
                                  equal_var=False)[1] < 0.05:
                    selectedFeatutesList.append(colName)
        if 'label' not in selectedFeatutesList: selectedFeatutesList = ['label'] + selectedFeatutesList
        # print(index)
        data1 = X_smote[X_smote['label'] == 0][selectedFeatutesList]
        data2 = X_smote[X_smote['label'] == 1][selectedFeatutesList]
        trainData = pd.concat([data1, data2])
        # print(trainData)
        trainData = shuffle(trainData)
        trainData.index = range(len(trainData))
        X = trainData[trainData.columns[1:]]
        y = trainData['label']
        alphas = np.logspace(-3, 1, 50)
        model_lassoCV = LassoCV(alphas=alphas, cv=5, max_iter=3000).fit(X, y)
        print(model_lassoCV.alpha_)
        coef = pd.Series(model_lassoCV.coef_, index=X.columns)
        index = coef[coef != 0].index
        X = X[index]
        print(coef[coef != 0].sort_values(axis=0, ascending=False))
        # featureNum = np.arange(len(index))
        # featureCoef = coef[coef != 0]
        # plt.bar(featureNum, featureCoef,
        #         color='lightblue',
        #         edgecolor='black',
        #         alpha=0.8)
        # plt.xticks(featureNum, index,
        #            rotation='45',
        #            ha='right',
        #            va='top')
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) +
              " variables")
        # print(X_Smote)
        # print(y_smote)
        return X, y

    @staticmethod
    def LDA_followed_PCA(data1, data2, data):
        selectedFeatutesList = []
        for colName in data.columns[8:]:
            selectedFeatutesList.append(colName)
        if 'label' not in selectedFeatutesList: selectedFeatutesList = ['label'] + selectedFeatutesList
        data1 = data1[selectedFeatutesList]
        data2 = data2[selectedFeatutesList]
        trainData = pd.concat([data1, data2])
        trainData = shuffle(trainData)
        trainData.index = range(len(trainData))
        X = trainData[trainData.columns[1:]]
        y = trainData['label']
        # X = X.fillna(0)
        # X = X.astype(np.float64)
        # X = StandardScaler().fit_transform(X)
        # X = pd.DataFrame(X)
        minmax_scale = preprocessing.MinMaxScaler().fit(X)
        normalX = minmax_scale.transform(X)
        # Principal Component Analysis
        pca = PCA(n_components=0.99)
        print(X)
        pca_fit_X = pca.fit(X)
        pca_fit = pca.transform(pca_fit_X)
        pca_fit = pd.DataFrame(pca_fit)
        # Linear Discriminant Analysis follwed by PCA
        lda = LinearDiscriminantAnalysis(n_components=4, solver='svd', shrinkage=None)
        X_features = lda.fit_transform(pca_fit, y)
        X_features = pd.DataFrame(X_features)
        X_features = np.array(X_features)
        return X_features, y

    @staticmethod
    def TTest_mRMR_svmRFE_selector(originData):
        selectedFeatutesList = []
        label = originData['label']
        colNames = originData[originData.columns[2:8]].columns
        data = originData[originData.columns[2:8]].fillna(0)
        data = data.astype(np.float64)
        data = StandardScaler().fit_transform(data)
        # minmax_scale = preprocessing.MinMaxScaler().fit(data)
        # data = minmax_scale.transform(data)
        data = pd.DataFrame(data)
        data.columns = colNames
        data['label'] = label
        # balanced Data
        smo = SMOTE(random_state=3)
        X_smote, y_smote = smo.fit_sample(data, data['label'])
        for colName in X_smote.columns[0:-1]:
            # if 'DWI' in colName:
            if levene(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[1] > 0.05 and \
                    ttest_ind(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[
                        1] < 0.05:
                selectedFeatutesList.append(colName)
            elif levene(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName])[1] <= 0.05 and \
                    ttest_ind(X_smote[X_smote['label'] == 0][colName], X_smote[X_smote['label'] == 1][colName],
                              equal_var=False)[1] < 0.05:
                selectedFeatutesList.append(colName)
        if 'label' not in selectedFeatutesList: selectedFeatutesList = ['label'] + selectedFeatutesList
        # print(index)
        data1 = X_smote[X_smote['label'] == 0][selectedFeatutesList]
        data2 = X_smote[X_smote['label'] == 1][selectedFeatutesList]
        trainData = pd.concat([data1, data2])
        # trainData = shuffle(trainData)
        # trainData.index = range(len(trainData))  # 打乱后重新标号
        X = trainData[trainData.columns[1:]]
        y = trainData['label']
        # print(X_Smote)
        # mRMR_features = pymrmr.mRMR(X_smote, 'MIQ', 15)
        # define MI_FS feature selection method
        feat_selector = mifs.MutualInformationFeatureSelector(method='JMIM')
        feat_selector.fit(X, y)
        # feat_selector._support_mask
        # feat_selector.ranking_
        # call transform() on X to filter it down to selected features
        # X_filtered = feat_selector.transform(X_smote)
        # X_filtered = pd.DataFrame(X_filtered)
        # print(feat_selector.ranking_)
        # if 'label' not in mRMR_features: mRMR_features = ['label'] + mRMR_features
        X_mRMR = X.loc[:, feat_selector._support_mask]
        colNames = X_mRMR.columns
        clf = LinearSVC()
        # featureNums = len(selectedFeatutesList)
        # print(featureNums)
        model = RFE(clf, n_features_to_select=len(feat_selector.ranking_))
        # print(y)
        # print(X_mRMR)
        model.fit(X_mRMR, y)
        feats = list(np.array(colNames)[model.support_])
        for featureNames in feats:
            print(featureNames)
        print(len(feats))
        X_RFE = X_mRMR[feats]
        return X_RFE, y

    @staticmethod
    def Pearson_RFE_selector(originData):
        label = originData['label']
        print(label[label[0:] == 0].values.size, label[label[0:] == 1].values.size)
        colNames = originData[originData.columns[2:-1]].columns
        data = originData[originData.columns[2:-1]].fillna(0)
        data = data.astype(np.float64)
        data = StandardScaler().fit_transform(data)
        data = pd.DataFrame(data)
        data.columns = colNames
        data['label'] = label
        # balanced Data
        smo = SMOTE(random_state=3)
        X_smote, y_smote = smo.fit_sample(data[data.columns[0:-1]], data['label'])
        # print(X_smote)
        # features_list = []
        # for colName in X_smote.columns:
        #     if 'DWI' in colName:
        #         features_list.append(colName)
        # X_smote = X_smote[features_list]
        # print(X_smote)
        # print(X_smote.corr(method='pearson').columns)
        # pearson_corr = X_smote.corr(method='pearson')
        # pearson_features_list = []
        # mean = pearson_corr['label'].mean()
        # for colName in pearson_corr[(abs(pearson_corr['label']) > mean) & (abs(pearson_corr['label']) < 1)].index:
        #     # if 'T2' in colName or 'ADC' in colName or 'DWI' in colName:
        #         pearson_features_list.append(colName)
        # print(X_smote[pearson_features_list])
        # X_pearson = X_smote[pearson_features_list]
        # y_pearson = X_smote['label']
        lr = LinearRegression()
        rfe = RFECV(lr, step=1, cv=5)
        rfe.fit(X_smote, y_smote)
        X_rfe = X_smote.loc[:, rfe.support_]
        # print(X_rfe)
        names = X_rfe.columns
        ranks = rfe.ranking_
        feature_indexes = []
        for i in range(len(ranks)):
            if ranks[i] == 1:
                feature_indexes += [i]
        print(len(feature_indexes))
        print(names)
        # print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
        return X_rfe, y_smote

    @staticmethod
    def train_classifier(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        print(len(X_train), len(X_test))
        # model_rf = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)
        # accuracy_rf = model_rf.score(X_test, y_test)
        # Cs = np.logspace(-1, 3, 10, base=2)
        # gammas = np.logspace(-4, 1, 50, base=2)
        # param_grid = dict(C=Cs, gamma=gammas)
        # grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=5).fit(X_train, y_train)
        # print(grid.best_params_)
        # C = grid.best_params_['C']
        # gamma = grid.best_params_['gamma']
        # pca = PCA(n_components=0.99)
        # pca.fit(X_train)
        # pca_fit_train = pca.transform(X_train)
        # pca_fit_test = pca.transform(X_test)
        # pca_train = pd.DataFrame(pca_fit_train)
        # pca_test = pd.DataFrame(pca_fit_test)
        best_auc = -1
        best_metric_epoch = 0
        rkf = RepeatedKFold(n_splits=5, n_repeats=1)
        class_modal_key = 't2+adc'
        # C_range = np.logspace(-2, 10, 15)
        # gamma_range = np.logspace(-10, 3, 15)
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        # grid.fit(X, y)
        #
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))
        # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
        #                                                      len(gamma_range))
        # plt.figure(figsize=(8, 6))
        # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
        #            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        # plt.xlabel('gamma')
        # plt.ylabel('C')
        # plt.colorbar()
        # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        # plt.yticks(np.arange(len(C_range)), C_range)
        # plt.title('Validation accuracy')
        # plt.show()
        val_acc = []
        val_auc = []
        for train_index, val_index in rkf.split(X_train):
            X_cv_train = X_train.iloc[train_index]
            X_val = X_train.iloc[val_index]
            y_cv_train = y_train.iloc[train_index]
            y_val = y_train.iloc[val_index]
            model_svm = svm.SVC(C=3727593.720314938, kernel='rbf', gamma=0.02275845926074791,
                                probability=True).fit(X_cv_train, y_cv_train)
            accuracy_svm = model_svm.score(X_val, y_val)
            y_probs = model_svm.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_probs[:, 1])
            val_acc.append(accuracy_svm)
            val_auc.append(auc)
        current_auc = sum(val_auc) / len(val_auc)
        current_acc = sum(val_acc) / len(val_acc)
        print(
            "current AUC: {:.4f} current accuracy: {:.4f}".format(
                current_auc, current_acc
            ))
        with open('../result/No2Hospital/binglishengji_model/best_clf_{classModal}.pickle'.format(
                classModal=class_modal_key),
                  'wb') as f:
            pickle.dump(model_svm, f)
        # print(pca_train.shape, pca_test.shape)
        # model_svm = svm.SVC(kernel='rbf', gamma=0.005, probability=True).fit(X_train, y_train)
        with open('../result/No2Hospital/binglishengji_model/best_clf_{classModal}.pickle'.format(classModal=class_modal_key),
                  'rb') as f:
            best_clf = pickle.load(f)
            y_train_probs = best_clf.predict_proba(X_train)
            train_auc = roc_auc_score(y_train, y_train_probs[:, 1])
            train_accuracy_rf = best_clf.score(X_train, y_train)
            print('train_auc = %.4f' % train_auc)
            print('train_accuracy = %.4f' % train_accuracy_rf)
            print(classification_report(y_train, best_clf.predict(X_train), digits=4))
            y_val_probs = best_clf.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, y_val_probs[:, 1])
            val_accuracy_rf = best_clf.score(X_val, y_val)
            print('val_auc = %.4f' % val_auc)
            print('val_accuracy = %.4f' % val_accuracy_rf)
            print(classification_report(y_val, best_clf.predict(X_val), digits=4))
            y_test_probs = best_clf.predict_proba(X_test)
            test_auc = roc_auc_score(y_test, y_test_probs[:, 1])
            test_accuracy_rf = best_clf.score(X_test, y_test)
            print('test_auc = %.4f' % test_auc)
            print('test_accuracy = %.4f' % test_accuracy_rf)
            print(classification_report(y_test, best_clf.predict(X_test), digits=4))
            fpr, tpr, thresholds = roc_curve(y_test, y_test_probs[:, 1])
            plt.plot(fpr, tpr, linewidth=2, label="ROC")
            plt.xlabel("false presitive rate")
            plt.ylabel("true presitive rate")
            plt.legend(loc=4)  # 图例的位置
            plt.savefig('../result/No2Hospital/binglishengji_roc/{classModal}_roc.png'.format(classModal=class_modal_key))
            # result_data = {'predict': y_test_probs[:, 1]}
            # result_data = pd.DataFrame(result_data)
            # result_data['label'] = y_test.iloc[:, ].values
            # print(result_data)
            # dca(result_data, 'label', 'predict')


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_best_gamma(X_train, y_train):
    model_svm = svm.SVC
    param_range = np.logspace(-6, -1, 100)
    train_loss, test_loss = validation_curve(
        model_svm(), X_train, y_train, param_name='gamma', param_range=param_range,
        cv=5, scoring='neg_mean_squared_error'
    )
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross_Validation')
    plt.xlabel('gamma ')
    plt.ylabel('Loss')
    plt.legend(loc='best')


def get_best_features_Nums(X_train, y_train, originNum):
    feature_num = 0
    best_acc = 0
    print(originNum)
    for i in range(originNum):
        clf = LinearSVC()
        model = RFE(clf, n_features_to_select=i + 1)
        model.fit(X_train, y_train)
        if model.score(X_train, y_train) > best_acc:
            best_acc = model.score(X_train, y_train)
            feature_num = i + 1
            print(best_acc)
    return feature_num


if __name__ == '__main__':
    # get_features()
    # fix_data_axis()
    # getNiiImg()
    x1, x2, mixData = load_data()
    trainX, trainY = ClassificationProcessing.TTest_Lasso(mixData)
    # trainX, trainY = ClassificationProcessing.LDA_followed_PCA(x1, x2, mixData)
    # trainX, trainY = ClassificationProcessing.TTest_mRMR_svmRFE_selector(mixData)
    # trainX, trainY = ClassificationProcessing.Pearson_RFE_selector(mixData)
    ClassificationProcessing.train_classifier(trainX, trainY)
