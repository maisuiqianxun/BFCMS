# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:35:19 2021

@author: maisui
"""

import numpy as np
import scipy.io as scio
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from functools import reduce
import operator
from pandas import DataFrame
from Add_HFCM import BFCMS, BFCMS_addHFCMS
from Add_SAE import BFCMS_addSAES

def translabel(trainlabel, NumberOfTrainSample):
    listlabel = list(np.squeeze(trainlabel))
    minlabel = int(np.min(listlabel))
    maxlabel = int(np.max(listlabel))
    transformedlabel = np.zeros((NumberOfTrainSample, maxlabel - minlabel + 1))
    # for i in range(100): # the most label number is set 100
    #     if not (i in listlabel):
    #         transformedlabel = np.zeros((NumberOfTrainSample, i))
    #         break
    for i in range(NumberOfTrainSample):
        location = int(trainlabel[i, 0])
        transformedlabel[i, location - minlabel] = 1
    return transformedlabel

def dataprocess(Train,Test):
    NumberOfTrainSample = Train.shape[0]
    NumberOfTestSample = Test.shape[0]
    featureDimension = Train.shape[1] - 1  # remove the dimension of the label " -1 "

    traindata = np.zeros((NumberOfTrainSample, featureDimension))
    trainlabel = np.zeros((NumberOfTrainSample, 1))
    for i in range(NumberOfTrainSample):
        trainlabel[i, 0] = Train.iat[i, 0]
        tmpArray = Train.iloc[[i]].values
        tmpArray = reduce(operator.add, tmpArray)
        minvalue = np.min(tmpArray[1:])
        maxvalue = np.max(tmpArray[1:])
        traindata[i, :] = (tmpArray[1:] - minvalue)/(maxvalue-minvalue)
    transformedtrainlabel = translabel(trainlabel, NumberOfTrainSample)

    testdata = np.zeros((NumberOfTestSample, featureDimension))
    testlabel = np.zeros((NumberOfTestSample, 1))
    for i in range(NumberOfTestSample):
        testlabel[i, 0] = Test.iat[i, 0]
        tmpArray = Test.iloc[[i]].values
        tmpArray = reduce(operator.add, tmpArray)
        minvalue = np.min(tmpArray[1:])
        maxvalue = np.max(tmpArray[1:])
        testdata[i, :] = (tmpArray[1:] - minvalue)/(maxvalue-minvalue)
    transformedtestlabel = translabel(testlabel, NumberOfTestSample)
    return traindata, transformedtrainlabel, testdata, transformedtestlabel


######################## load time series data with benchmark datasets ##############################
sptio = 0.7  #  # a para used to divide validation from train data 

N1 = 6 # default 60  #  # the length of each window
N2 = 25 # default 20 #  # the number of windows -------Feature mapping layer
N3 = 1 # default 7 #  # of HFCMs nodes -----Enhance HFCM layer
order = [3] #  # of order for each HFCM node -----Enhance HFCM layer


M1 = 50  #  # of adding enhance nodes

s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

root_dir = r'D:\code\python\BFCMS_code\data_TSC'

trainAccuracySet = []
trainTimeSet = []
testAccuracySet = []
testTimeSet = []
dataset_name = []
RatioSetlist = []
datasetlist = []
for file in os.listdir(root_dir):
    datasetlist.append(file)
    print(file)
    file_name = root_dir + '/' + file
    for file1 in os.listdir(file_name):
        if file1 == file + "_TRAIN.tsv":
            file_name1 = file_name + '/' + file1
            Train = pd.read_csv(file_name1, delimiter='\t', header=None)
        elif file1 == file + "_TEST.tsv":
            file_name1 = file_name + '/' + file1
            Test = pd.read_csv(file_name1, delimiter='\t', header=None)
    # #######################training time cost with different ratio########################
    # TrainTimeCostlistWithDifferentRatio = []
    # for ratio in np.arange(0.1, 1.1, 0.1).tolist():
    #     numsamples = len(Train)
    #     Train1 = Train.iloc[:int(numsamples * ratio), :]
    #     traindata, transformedtrainlabel, testdata, transformedtestlabel = dataprocess(Train1,Test)
    #
    #     print('-------------------BFCMS_BASE---------------------------')
    #
    #     train_acc, test_acc, traintime = BFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order)
    #     print(train_acc,'____________', test_acc)
    #     TrainTimeCostlistWithDifferentRatio.append(traintime)
    # print(TrainTimeCostlistWithDifferentRatio)
    # RatioSetlist.append(TrainTimeCostlistWithDifferentRatio)
    traindata, transformedtrainlabel, testdata, transformedtestlabel = dataprocess(Train, Test)

    print('-------------------BFCMS_BASE---------------------------')
    train_acc, test_acc, traintime = BFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order)
    print(train_acc,'____________', test_acc)

# Result2 = {'Dataset':datasetlist, 'RatioAcc':RatioSetlist }
# df2 = DataFrame(Result2)
# df2.sort_values(by=['Dataset'])
# print(df2)
# df2.to_csv("BFCMS_timecost.tsv", sep='\t')

    # # #######################the Parameter analysis: order start########################
    # orderset = [[1],[2],[3],[4],[5],[6],[7],[8]]
    # AccCororder = []
    # for order in orderset:
    #     train_acc, test_acc = BFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order)
    #     AccCororder.append(train_acc)
    # x = np.arange(1, len(orderset) + 1, 1)
    # y = np.array(AccCororder)
    # xnew = np.linspace(1,len(orderset),len(orderset) * 10)
    # intfunc = interpolate.interp1d(x,y,fill_value="extrapolate")
    # ynew = intfunc(xnew)    
    # import seaborn as sns
    # plt.style.use(['ggplot','seaborn-paper'])
    # fig4 = plt.figure()
    # ax41 = fig4.add_subplot(111)
    # font1 = {'weight' : 'normal',
    # 'size'   : 20,
    # }
    # ax41.plot(x, y, 'D', color = 'blue')
    # ax41.plot(xnew, ynew, '--', color = 'r', linewidth=1)
    # plt.xticks(fontsize = 20)
    # plt.yticks(fontsize = 20)
    # ax41.set_ylabel("accuracy", font1)
    # ax41.set_xlabel('$L$', font1)
    # plt.title(file, font1)
    # plt.tight_layout()
    # if not os.path.exists(r"./ParameterVisualization"):
    #     os.makedirs(r"./ParameterVisualization")
    # plt.savefig(r"./ParameterVisualization/%s.tiff" % (file + "order"),dpi = 300)    
    # # #######################the Parameter analysis: order end########################
    
    # # #######################the Parameter analysis: order start########################
    # AccCororder = []
    # N1set = [3, 4, 5, 6, 7, 8, 9, 10]
    # for N1 in N1set:
    #     train_acc, test_acc = BFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order)
    #     AccCororder.append(train_acc)
    # x = np.arange(3, len(N1set) +3, 1)
    # y = np.array(AccCororder)   
    # import seaborn as sns
    # plt.style.use(['ggplot','seaborn-paper'])
    # fig4 = plt.figure()
    # ax41 = fig4.add_subplot(111)
    # font1 = {'weight' : 'normal',
    # 'size'   : 20,
    # }
    # ax41.plot(x, y, 'D', color = 'blue')
    # ax41.plot(x, y, '--', color = 'r', linewidth=1)
    # plt.xticks(fontsize = 20)
    # plt.yticks(fontsize = 20)
    # ax41.set_ylabel("accuracy", font1)
    # ax41.set_xlabel('$k$', font1)
    # plt.title(file, font1)
    # plt.tight_layout()
    # if not os.path.exists(r"./ParameterVisualization"):
    #     os.makedirs(r"./ParameterVisualization")
    # plt.savefig(r"./ParameterVisualization/%s.tiff" % (file + "N1"),dpi = 300)    
    # # #######################the Parameter analysis: order end########################
    
    
    # #######################the effects of adding HFCMs: start########################
    # print('-------------------BFCMS_addHFCMS------------------------')
    # NumAddedHFCMs = 10
    # train_acc2, test_acc2 = BFCMS_addHFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order, NumAddedHFCMs)
    # print(train_acc2,'____________', test_acc2)
    # x = np.arange(1, NumAddedHFCMs + 1, 1)
    # y = np.array(train_acc2)
    # xnew = np.linspace(1,NumAddedHFCMs,NumAddedHFCMs * 10)
    # intfunc = interpolate.interp1d(x,y,fill_value="extrapolate")
    # ynew = intfunc(xnew)    
    # import seaborn as sns
    # plt.style.use(['ggplot','seaborn-paper'])
    # fig4 = plt.figure()
    # ax41 = fig4.add_subplot(111)
    # font1 = {'weight' : 'normal',
    # 'size'   : 20,
    # }
    # ax41.plot(x, y, 'D', color = '#f97306')
    # ax41.plot(xnew, ynew, '--', color = 'b', linewidth=1)
    # plt.xticks(fontsize = 20)
    # plt.yticks(fontsize = 20)
    # ax41.set_ylabel("accuracy", font1)
    # ax41.set_xlabel('$m_a$', font1)
    # plt.title(file, font1)
    # plt.tight_layout()
    # if not os.path.exists(r"./Outcome_for_papers"):
    #     os.makedirs(r"./Outcome_for_papers")
    # plt.savefig(r"./Outcome_for_papers/%s.tiff" % (file + "HFCM"),dpi = 300)
    # #######################the effects of adding HFCMs: end########################
    
    
    # #######################the effects of adding SAEs: start########################
    # print('-------------------BFCMS_addSAES------------------------')
    # NumAddedSAEs = 10
    # train_acc3, test_acc3 = BFCMS_addSAES(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1, N2, N3, order, NumAddedSAEs)
    # print(train_acc3,'____________', test_acc3)
    # x = np.arange(1, NumAddedSAEs + 1, 1)
    # y = np.array(train_acc3)
    # xnew = np.linspace(1,NumAddedSAEs, NumAddedSAEs * 10)
    # intfunc = interpolate.interp1d(x,y,fill_value="extrapolate")
    # ynew = intfunc(xnew)    
    # import seaborn as sns
    # plt.style.use(['ggplot','seaborn-paper'])

    # fig4 = plt.figure()
    # ax41 = fig4.add_subplot(111)
    # font1 = {'weight' : 'normal',
    # 'size'   : 20,
    # }
    # ax41.plot(x, y, 'D', color = '#f97306')
    # ax41.plot(xnew, ynew, '--', color = 'b', linewidth=1)
    # plt.xticks(fontsize = 20)
    # plt.yticks(fontsize = 20)
    # ax41.set_ylabel("accuracy", font1)
    # ax41.set_xlabel('$n_a$', font1)
    # plt.title(file, font1)
    # plt.tight_layout()
    # if not os.path.exists(r"./Outcome_for_papers"):
    #     os.makedirs(r"./Outcome_for_papers")
    # plt.savefig(r"./Outcome_for_papers/%s.tiff" % (file + "SAE"),dpi = 300)
    # #######################the effects of adding SAEs: end########################
    
    
    
    
    # # find the optimal parameters
    # maxtrainacc = 0
    # maxtestacc = 0
    # # N1 is the length of each feature series; N2 is the number of SAEs, N3 denotes the number of HFCMs
    # for N1 in range(10, 100, 5): # [15, 30, 45]:
    #     for N2 in range(3, 6, 1): # [20]: # [10, 15, 20, 25, 30]
    #         for N3 in range(1, 2, 1):
    #             for order in [[1], [2], [3], [4]]:
    #                 train_acc, test_acc, traintime = BFCMS(traindata, transformedtrainlabel, testdata, transformedtestlabel, s, C, N1,
    #                                             N2, N3, order)
    #
    #
    #                 if test_acc > maxtestacc:
    #                     maxtrainacc = train_acc
    #                     maxtestacc = test_acc
    #                     optiparas = [N1, N2, N3, order[0]]
    #                     print(train_acc, test_acc, optiparas)
    # print('Training accurate is', maxtrainacc * 100, '%')
    # print('Testing accurate is', maxtestacc * 100, '%')
    # print('opt parameter combinations', optiparas)
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    