import os
import csv


def writeInCsv(inputList, inputType, csvFile):
    for fileName in inputList:
        # if len(fileName.split('.')[0].split('-')) > 3:
        #     inputName = os.path.splitext(fileName)[0].split('-')[1] + os.path.splitext(fileName)[0].split('-')[2] \
        #                 + os.path.splitext(fileName)[0].split('-')[3]
        #     if '.' in inputName:
        #         inputName = inputName.split('.')[0] + inputName.split('.')[1]
        #     # print(inputName)
        # else:
        #     print(fileName)
        #     inputName = os.path.splitext(fileName)[0].split('-')[1] + os.path.splitext(fileName)[0].split('-')[2]
        #     if '.' in inputName:
        #         inputName = inputName.split('.')[0] + inputName.split('.')[1]
        head = [fileName, inputType]
        writer = csv.writer(csvFile)
        # 写入多行数据
        writer.writerow(head)
