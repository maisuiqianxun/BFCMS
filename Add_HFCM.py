# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 21:48:23 2021

@author: maisui
"""
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
def sigmoid(data):
    return 1.0/(1+np.exp(-data))

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))

def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z

def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

def BFCMS(train_x, train_y,test_x,test_y, s, C, N1, N2, N3, order):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    time_start=time.time()#计时开始
    FeatureTimeSeriesSet = []
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        FeatureTimeSeriesSet.append(outputOfEachWindow)
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    # Another form of feature time series for all train samples
    FeatureTimeSeriesSet2 = []
    for i in range(train_x.shape[0]):
        tmp = np.zeros((N2, N1))
        for j in range(N2):
            tmp[j, :] = FeatureTimeSeriesSet[j][i, :]
        FeatureTimeSeriesSet2.append(tmp)

    # Input form adjustment of high order FCMs
    InputOfHFCMSet = []
    for i in range(train_x.shape[0]):
        tmp = np.zeros((N1 - order[0], N2 * order[0]))
        InnerIndex = 0
        for j in range(N1 - order[0]):
            for m in range(order[0]):
                tmp[j, m * N2:(m + 1) * N2] = np.transpose(FeatureTimeSeriesSet2[i][:, InnerIndex])
                InnerIndex = InnerIndex + 1
            InnerIndex = InnerIndex - order[0] + 1
        tmpAddBias = np.hstack([tmp, 0.1 * np.ones((tmp.shape[0], 1))])
        InputOfHFCMSet.append(tmpAddBias)

    # initialize the weights of HFCM(setting in paper TFS)
    weightOfHFCM = []
    for i in range(N3):
        random.seed(67797325 - i)
        ithweightOfHFCM = 2 * random.randn(N2 * order[0] + 1, N2) - 1
        weightOfHFCM.append(ithweightOfHFCM)
    
    
    # calculate the output of HFCM
    OutputOfHFCMset = []
    for i in range(train_x.shape[0]):
        OutputOfAllHFCMs = np.array(0)
        for j in range(len(weightOfHFCM)):
            OutputOfjthHFCM = sigmoid(np.dot(InputOfHFCMSet[i], weightOfHFCM[j])).reshape(-1)
            OutputOfAllHFCMs = np.hstack((OutputOfAllHFCMs, OutputOfjthHFCM))
        OutputOfHFCMset.append(OutputOfAllHFCMs[1:])
    
    OutputOfEnhanceLayer = OutputOfHFCMset[0]
    for i in range(1, len(OutputOfHFCMset)):
        OutputOfEnhanceLayer = np.vstack((OutputOfEnhanceLayer, OutputOfHFCMset[i]))


    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, C)
    OutputWeight = np.dot(pinvOfInput,train_y) 
    time_end=time.time() 
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    # print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is of is: %f' % trainTime, 's')
    
    
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()
    
    FeatureTimeSeriesTestSet = []   
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        outputOfEachWindowTest = (ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = outputOfEachWindowTest
        FeatureTimeSeriesTestSet.append(outputOfEachWindowTest)
        
    # Another form of feature time series for all test samples
    FeatureTimeSeriesTestSet2 = []
    for i in range(test_x.shape[0]):
        tmp = np.zeros((N2, N1))
        for j in range(N2):
            tmp[j, :] = FeatureTimeSeriesTestSet[j][i, :]
        FeatureTimeSeriesTestSet2.append(tmp)

    # Input test form adjustment of high order FCMs
    InputOfHFCMSetTest = []
    for i in range(test_x.shape[0]):
        tmp = np.zeros((N1 - order[0], N2 * order[0]))
        InnerIndex = 0
        for j in range(N1 - order[0]):
            for m in range(order[0]):
                tmp[j, m * N2:(m + 1) * N2] = np.transpose(FeatureTimeSeriesTestSet2[i][:, InnerIndex])
                InnerIndex = InnerIndex + 1
            InnerIndex = InnerIndex - order[0] + 1
        tmpAddBias = np.hstack([tmp, 0.1 * np.ones((tmp.shape[0], 1))])
        InputOfHFCMSetTest.append(tmpAddBias)
        
    
    # calculate the output of HFCM
    OutputOfHFCMsetTest = []
    for i in range(test_x.shape[0]):
        OutputOfAllHFCMsTest = np.array(0)
        for j in range(len(weightOfHFCM)):
            OutputOfjthHFCM = sigmoid(np.dot(InputOfHFCMSetTest[i], weightOfHFCM[j])).reshape(-1)
            OutputOfAllHFCMsTest = np.hstack((OutputOfAllHFCMsTest, OutputOfjthHFCM))
        OutputOfHFCMsetTest.append(OutputOfAllHFCMs[1:])
    
    OutputOfEnhanceLayerTest = OutputOfHFCMsetTest[0]
    for i in range(1, len(OutputOfHFCMsetTest)):
        OutputOfEnhanceLayerTest = np.vstack((OutputOfEnhanceLayerTest, OutputOfHFCMsetTest[i]))
        

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    # print('Testing accurate is' ,testAcc * 100,'%')
    print('Test time is of is: %f' % testTime, 's')

    return trainAcc, testAcc, trainTime

def BFCMS_addHFCMS(train_x, train_y,test_x,test_y, s, c, N1, N2, N3, order, NumAddedHFCMs):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2*N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    time_start=time.time()#计时开始
    FeatureTimeSeriesSet = []
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) 
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1*i:N1*(i+1)] = outputOfEachWindow
        FeatureTimeSeriesSet.append(outputOfEachWindow)
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    # Another form of feature time series for all train samples
    FeatureTimeSeriesSet2 = []
    for i in range(train_x.shape[0]):
        tmp = np.zeros((N2, N1))
        for j in range(N2):
            tmp[j, :] = FeatureTimeSeriesSet[j][i, :]
        FeatureTimeSeriesSet2.append(tmp)

    # Input form adjustment of high order FCMs
    InputOfHFCMSet = []
    for i in range(train_x.shape[0]):
        tmp = np.zeros((N1 - order[0], N2 * order[0]))
        InnerIndex = 0
        for j in range(N1 - order[0]):
            for m in range(order[0]):
                tmp[j, m * N2:(m + 1) * N2] = np.transpose(FeatureTimeSeriesSet2[i][:, InnerIndex])
                InnerIndex = InnerIndex + 1
            InnerIndex = InnerIndex - order[0] + 1
        tmpAddBias = np.hstack([tmp, 0.1 * np.ones((tmp.shape[0], 1))])
        InputOfHFCMSet.append(tmpAddBias)

    # initialize the weights of HFCM(setting in paper TFS)
    weightOfHFCM = []
    for i in range(N3):
        random.seed(67797325 - i)
        ithweightOfHFCM = 2 * random.randn(N2 * order[0] + 1, N2) - 1
        weightOfHFCM.append(ithweightOfHFCM)
    
    
    # calculate the output of HFCM
    OutputOfHFCMset = []
    for i in range(train_x.shape[0]):
        OutputOfAllHFCMs = np.array(0)
        for j in range(len(weightOfHFCM)):
            OutputOfjthHFCM = sigmoid(np.dot(InputOfHFCMSet[i], weightOfHFCM[j])).reshape(-1)
            OutputOfAllHFCMs = np.hstack((OutputOfAllHFCMs, OutputOfjthHFCM))
        OutputOfHFCMset.append(OutputOfAllHFCMs[1:])
    
    OutputOfEnhanceLayer = OutputOfHFCMset[0]
    for i in range(1, len(OutputOfHFCMset)):
        OutputOfEnhanceLayer = np.vstack((OutputOfEnhanceLayer, OutputOfHFCMset[i]))


    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput,train_y) 
    time_end=time.time() 
    trainTime = time_end - time_start
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    
    
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()
    
    FeatureTimeSeriesTestSet = []   
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        outputOfEachWindowTest = (ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = outputOfEachWindowTest
        FeatureTimeSeriesTestSet.append(outputOfEachWindowTest)
        
    # Another form of feature time series for all test samples
    FeatureTimeSeriesTestSet2 = []
    for i in range(test_x.shape[0]):
        tmp = np.zeros((N2, N1))
        for j in range(N2):
            tmp[j, :] = FeatureTimeSeriesTestSet[j][i, :]
        FeatureTimeSeriesTestSet2.append(tmp)

    # Input test form adjustment of high order FCMs
    InputOfHFCMSetTest = []
    for i in range(test_x.shape[0]):
        tmp = np.zeros((N1 - order[0], N2 * order[0]))
        InnerIndex = 0
        for j in range(N1 - order[0]):
            for m in range(order[0]):
                tmp[j, m * N2:(m + 1) * N2] = np.transpose(FeatureTimeSeriesTestSet2[i][:, InnerIndex])
                InnerIndex = InnerIndex + 1
            InnerIndex = InnerIndex - order[0] + 1
        tmpAddBias = np.hstack([tmp, 0.1 * np.ones((tmp.shape[0], 1))])
        InputOfHFCMSetTest.append(tmpAddBias)
        
    
    # calculate the output of HFCM
    OutputOfHFCMsetTest = []
    for i in range(test_x.shape[0]):
        OutputOfAllHFCMsTest = np.array(0)
        for j in range(len(weightOfHFCM)):
            OutputOfjthHFCM = sigmoid(np.dot(InputOfHFCMSetTest[i], weightOfHFCM[j])).reshape(-1)
            OutputOfAllHFCMsTest = np.hstack((OutputOfAllHFCMsTest, OutputOfjthHFCM))
        OutputOfHFCMsetTest.append(OutputOfAllHFCMs[1:])
    
    OutputOfEnhanceLayerTest = OutputOfHFCMsetTest[0]
    for i in range(1, len(OutputOfHFCMsetTest)):
        OutputOfEnhanceLayerTest = np.vstack((OutputOfEnhanceLayerTest, OutputOfHFCMsetTest[i]))
        

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc * 100,'%')
    print('Testing time is ',testTime,'s')
    
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    trainAccSet = []
    testAccSet = []
    for e in list(range(NumAddedHFCMs)):
        random.seed(e)
        AddedweightOfHFCM = 2 * np.random.randn(N2 * order[0] + 1, N2) - 1
        
        # calculate the output of added HFCM
        AddedOutputOfHFCMset = []
        for i in range(train_x.shape[0]):
            AddedOutputOfHFCM = sigmoid(np.dot(InputOfHFCMSet[i], AddedweightOfHFCM)).reshape(-1)
            AddedOutputOfHFCMset.append(AddedOutputOfHFCM)
        
        AddedOutputOfEnhanceLayer = AddedOutputOfHFCMset[0]
        for i in range(1, len(AddedOutputOfHFCMset)):
            AddedOutputOfEnhanceLayer = np.vstack((AddedOutputOfEnhanceLayer, AddedOutputOfHFCMset[i]))
        
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,AddedOutputOfEnhanceLayer])

        D = pinvOfInput.dot(AddedOutputOfEnhanceLayer)
        C = AddedOutputOfEnhanceLayer - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        # train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        # train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        trainAccSet.append(TrainingAccuracy)
        

        time_start = time.time()
        
         # calculate the output of added HFCM for test
        AddedOutputOfHFCMsetTest = []
        for i in range(test_x.shape[0]):
            AddedOutputOfHFCMTest = sigmoid(np.dot(InputOfHFCMSetTest[i], AddedweightOfHFCM)).reshape(-1)
            AddedOutputOfHFCMsetTest.append(AddedOutputOfHFCMTest)
            
        AddedOutputOfEnhanceLayerTest = AddedOutputOfHFCMsetTest[0]
        for i in range(1, len(AddedOutputOfHFCMsetTest)):
            AddedOutputOfEnhanceLayerTest = np.vstack((AddedOutputOfEnhanceLayerTest, AddedOutputOfHFCMsetTest[i]))
        

        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, AddedOutputOfEnhanceLayerTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        testAccSet.append(TestingAcc)
        
        Test_time = time.time() - time_start
        # test_time[0][e+1] = Test_time
        # test_acc[0][e+1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )


    return trainAccSet, testAccSet