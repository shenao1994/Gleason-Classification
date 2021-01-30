import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.metrics import cohen_kappa_score


def statistics_method():
    Accuracy_list = []
    Kappa_list = []
    # AUC_list = []
    # Sensitivity_list = []
    # Specificity_list = []

    for num in range(1, 6):
        csvFilePath = '../result/GLeason/2d_output/t2+adc_2d_one-hot_SPP_model{num}.csv'.format(num=num)
        labelList = []
        totalPredList = []
        outputList = []
        infoList = []
        patientsList = []
        with open(csvFilePath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, dialect='excel')
            for row in reader:
                infoList.append(row[0].split('\\')[-1].split('-')[1] + '+' + row[1])
                if row[0].split('\\')[-1].split('-')[1] not in patientsList:
                    patientsList.append(row[0].split('\\')[-1].split('-')[1])
                    # labelList.append(0 if int(row[0].split('.')[0][-1]) < 1 else 1)
                    labelList.append(int(row[0].split('.')[0][-1]) - 1)
        for patientName in patientsList:
            predList = []
            for info in infoList:
                if patientName in info:
                    predList.append(float(info.split('+')[1]))
            # totalPredList.append(sum(predList) / len(predList)) # Binary Classification
            totalPredList.append(max(predList))  # Gleason Classification
        # Gleason Classification
        # print(totalPredList)
        # for i in range(len(totalPredList)):
        #     totalPredList[i] = int(totalPredList[i] * 4)
        print(totalPredList)
        print(labelList)
        Accuracy_list.append(accuracy_score(labelList, totalPredList))
        kappa_value = cohen_kappa_score(labelList, totalPredList, weights='quadratic')
        Kappa_list.append(kappa_value)
    print(Kappa_list)
    print(Accuracy_list)
    print('Cross_Validation Kappa:%.5f' % (sum(Kappa_list) / len(Kappa_list)))
    print('Cross_Validation Accuracy:%.5f' % (sum(Accuracy_list) / len(Accuracy_list)))
    # Binary Classification
    #     fpr, tpr, thresholds = roc_curve(labelList, totalPredList)
    #     roc_auc = auc(fpr, tpr)
    #     AUC_list.append(roc_auc)
    #     for i in range(len(totalPredList)):
    #         if totalPredList[i] < 0.5:
    #             totalPredList[i] = 0
    #         else:
    #             totalPredList[i] = 1
    #     Accuracy_list.append(accuracy_score(labelList, totalPredList))
    #     plt.plot(fpr, tpr, linewidth=2, label="ROC")
    #     plt.xlabel("false presitive rate")
    #     plt.ylabel("true presitive rate")
    #     plt.legend(loc=4)  # 图例的位置
    #     plt.show()
    #     sensitivity = recall_score(labelList, totalPredList)
    #     specificity = accuracy_score(labelList, totalPredList) * 2 - sensitivity
    #     Sensitivity_list.append(sensitivity)
    #     Specificity_list.append(specificity)
    # print(AUC_list)
    # print(Accuracy_list)
    # print(Sensitivity_list)
    # print(Specificity_list)
    # print('Cross_Validation AUC:%.5f' % (sum(AUC_list) / len(AUC_list)))
    # print('Cross_Validation Accuracy:%.5f' % (sum(Accuracy_list) / len(Accuracy_list)))
    # print('Cross_Validation Sensitivity:%.5f' % (sum(Sensitivity_list) / len(Sensitivity_list)))
    # print('Cross_Validation Specificity:%.5f' % (sum(Specificity_list) / len(Specificity_list)))


if __name__ == '__main__':
    statistics_method()
