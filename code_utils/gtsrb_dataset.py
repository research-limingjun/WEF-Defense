"""
======================
@author:Mr.li
@time:2022/3/15:18:42
@email:1035626671@qq.com
======================
"""
# !usr/bin/env python
# encoding:utf-8
from __future__ import division



import os
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convertTrainData(dataDir='data/GTSRB/Final_Training/Images/', saveDir='dataset/train/'):

    labelDir = os.listdir(dataDir)
    for label in labelDir:
        oneDir = dataDir + label + '/'
        oneSaveDir = saveDir + label + '/'
        if not os.path.exists(oneSaveDir):
            os.makedirs(oneSaveDir)
        for one_file in os.listdir(oneDir):
            if one_file.endswith(".csv"):
                csvFile = os.path.join(oneDir, one_file)
        csv_data = pd.read_csv(csvFile)
        csv_data_array = np.array(csv_data)
        for i in range(csv_data_array.shape[0]):
            csv_data_list = np.array(csv_data)[i, :].tolist()[0].split(";")
            one_ppm = os.path.join(oneDir, csv_data_list[0])
            img = PIL.Image.open(one_ppm)
            box = [int(csv_data_list[3]), int(csv_data_list[4]), int(csv_data_list[5]), int(csv_data_list[6])]
            img = img.crop(box)
            one_save_path = oneSaveDir + str(len(os.listdir(oneSaveDir)) + 1) + '.png'
            img.save(one_save_path)


def convertTestData(dataDir='data/GTSRB/Final_Test/Images/', saveDir='dataset/test/'):

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    file_list = os.listdir(dataDir)
    for one_file in file_list:
        if one_file.endswith(".csv"):
            csvFile = os.path.join(dataDir, one_file)
    csv_data = pd.read_csv(csvFile)
    csv_data_array = np.array(csv_data)
    for i in range(csv_data_array.shape[0]):
        csv_data_list = np.array(csv_data)[i, :].tolist()[0].split(";")
        one_ppm = os.path.join(dataDir, csv_data_list[0])
        img = PIL.Image.open(one_ppm)
        box = [int(csv_data_list[3]), int(csv_data_list[4]), int(csv_data_list[5]), int(csv_data_list[6])]
        img = img.crop(box)
        one_save_path = saveDir + str(len(os.listdir(saveDir)) + 1) + '.png'
        img.save(one_save_path)


if __name__ == "__main__":
    convertTrainData(dataDir='./data/GTSRB/Final_Training/Images/', saveDir='./data/GTSRB/train/')

    convertTestData(dataDir='./data/GTSRB/Final_Test/Images/', saveDir='./data/GTSRB/test/')
