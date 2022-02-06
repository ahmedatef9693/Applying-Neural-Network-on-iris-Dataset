#from Task1Gui import *
import pandas as pd
import numpy as np
import random as rn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt




Data = pd.read_csv('IrisData.txt',',',names = ['X1','X2','X3','X4','Class'])
Data = Data.drop(Data.index[0])
#Label Encoding
NewDataSet = { "Class":{"Iris-setosa":1,"Iris-versicolor":0,"Iris-virginica":-1} }
Data.replace(NewDataSet , inplace = True)





#Shuffling Then Getting Thirty Percent
def ShufflingClasses(Class1,Class2,Bias):
    global Data
    FirstClass = pd.DataFrame()
    SecondClass = pd.DataFrame()
    FirstThirtyTrainC1 = []
    FirstThirtyTrainC2 = []
    FirstTwentyTestC1 = []
    FirstTwentyTestC2 = []
       #(1,2) or (2,1)
    if ((Class1 =='Iris-setosa') and (Class2 == 'Iris-versicolor')) or((Class1 =='Iris-versicolor') and (Class2 == 'Iris-setosa')):
        #Training Data
        FirstClass = shuffle(Data.iloc[0:50,:])
        FirstClass.reset_index(drop = True , inplace=True)
        FirstThirtyTrainC1 = np.array(FirstClass.iloc[0:30,:])
        FirstThirtyTrainC1 = np.ndarray.astype(FirstThirtyTrainC1,float)

        SecondClass = shuffle(Data.iloc[50:100,:])
        SecondClass.reset_index(drop = True , inplace=True)
        FirstThirtyTrainC2 = np.array(SecondClass.iloc[0:30,:])
        FirstThirtyTrainC2 = np.ndarray.astype(FirstThirtyTrainC2,float)


        #Testing Data
        FirstTwentyTestC1 = np.array(FirstClass.iloc[30:50,:])
        FirstTwentyTestC1 = np.ndarray.astype(FirstTwentyTestC1, float)

        FirstTwentyTestC2 = np.array(SecondClass.iloc[30:50, :])
        FirstTwentyTestC2 = np.ndarray.astype(FirstTwentyTestC2, float)


        #(1,3) or (3,1)
    elif ((Class1 =='Iris-setosa') and (Class2 == 'Iris-virginica')) or((Class1 =='Iris-virginica') and (Class2 == 'Iris-setosa')):
        #Train
        FirstClass = shuffle(Data.iloc[0:50, :])
        FirstClass.reset_index(drop=True, inplace=True)
        FirstThirtyTrainC1 = np.array(FirstClass.iloc[0:30, :])
        FirstThirtyTrainC1 = np.ndarray.astype(FirstThirtyTrainC1, float)


        SecondClass = shuffle(Data.iloc[100:150, :])
        SecondClass.reset_index(drop=True, inplace=True)
        FirstThirtyTrainC2 = np.array(SecondClass.iloc[0:30, :])
        FirstThirtyTrainC2 = np.ndarray.astype(FirstThirtyTrainC2, float)

        #Test
        FirstTwentyTestC1 = np.array(FirstClass.iloc[30:50, :])
        FirstTwentyTestC1 = np.ndarray.astype(FirstTwentyTestC1, float)

        FirstTwentyTestC2 = np.array(SecondClass.iloc[30:50, :])
        FirstTwentyTestC2 = np.ndarray.astype(FirstTwentyTestC2, float)


        #(2,3) or (3,2)
    elif (Class1 =='Iris-versicolor') and (Class2 == 'Iris-virginica') or((Class1 =='Iris-virginica') and (Class2 == 'Iris-versicolor')):
        #Train
        FirstClass = shuffle(Data.iloc[50:100, :])
        FirstClass.reset_index(drop=True, inplace=True)
        FirstThirtyTrainC1 = np.array(FirstClass.iloc[0:30, :])
        FirstThirtyTrainC1 = np.ndarray.astype(FirstThirtyTrainC1, float)

        SecondClass = shuffle(Data.iloc[100:150, :])
        SecondClass.reset_index(drop=True, inplace=True)
        FirstThirtyTrainC2 = np.array(SecondClass.iloc[0:30, :])
        FirstThirtyTrainC2 = np.ndarray.astype(FirstThirtyTrainC2, float)

        #Test
        FirstTwentyTestC1 = np.array(FirstClass.iloc[30:50, :])
        FirstTwentyTestC1 = np.ndarray.astype(FirstTwentyTestC1, float)

        FirstTwentyTestC2 = np.array(SecondClass.iloc[30:50, :])
        FirstTwentyTestC2 = np.ndarray.astype(FirstTwentyTestC2, float)
    return FirstThirtyTrainC1,FirstThirtyTrainC2,FirstTwentyTestC1,FirstTwentyTestC2






def Signum(Entry):
    if Entry > 0:
        return 1
    elif Entry == 0:
        return 0
    elif Entry < 0:
        return -1




def GenerateFinalInputMatrix(FirstFeatureMatrix,SecondFeatureMatrix,Bias):
    #in this function we are concatenating bias value if it exist with first feature and second feature
    FinalMatrix = np.empty([60,3])
    BiasValue = 0
    if Bias == 1:
        BiasValue = 1
    elif Bias == 0:
        BiasValue = 0
    for row in range(60):
        FinalMatrix[row][0] = BiasValue
        FinalMatrix[row][1] = FirstFeatureMatrix[row]
        FinalMatrix[row][2] = SecondFeatureMatrix[row]
    return FinalMatrix



#Testing Matrix
def GenerateTestInput(FirstFeatureMatrix,SecondFeatureMatrix,Bias):
    TestMatrix = np.empty([40,3])
    BiasValue = 0
    if Bias == 1:
        BiasValue = 1
    elif Bias == 0:
        BiasValue = 0
    for row in range(40):
        TestMatrix[row][0] = BiasValue
        TestMatrix[row][1] = FirstFeatureMatrix[row]
        TestMatrix[row][2] = SecondFeatureMatrix[row]
    return TestMatrix


def GenerateWeights():
    WeightMatrix = np.empty([3,1],float)
    for i in range(3):
        WeightMatrix[i] = rn.random()
    return WeightMatrix


def SelectFeatures(InitializedInputMatrix,InitializedInputTestMatrix,F1,F2,FlagTrainOrTest):
    MyFirstFeatureList = []
    MySecondFeatureList = []
    YExact = []
    if FlagTrainOrTest == 0: #Which Means We Will Train
        if F1 == 'X1':
            MyFirstFeatureList.append(InitializedInputMatrix[:, 0])
        elif F1 == 'X2':
            MyFirstFeatureList.append(InitializedInputMatrix[:, 1])
        elif F1 == 'X3':
            MyFirstFeatureList.append(InitializedInputMatrix[:, 2])
        elif F1 == 'X4':
            MyFirstFeatureList.append(InitializedInputMatrix[:, 3])
        if F2 == 'X1':
            MySecondFeatureList.append(InitializedInputMatrix[:, 0])
        elif F2 == 'X2':
            MySecondFeatureList.append(InitializedInputMatrix[:, 1])
        elif F2 == 'X3':
            MySecondFeatureList.append(InitializedInputMatrix[:, 2])
        elif F2 == 'X4':
            MySecondFeatureList.append(InitializedInputMatrix[:, 3])
        YExact.append(InitializedInputMatrix[:, 4])
    elif FlagTrainOrTest ==1: # Which Means We Will Test
        if F1 == 'X1':
            MyFirstFeatureList.append(InitializedInputTestMatrix[:, 0])
        elif F1 == 'X2':
            MyFirstFeatureList.append(InitializedInputTestMatrix[:, 1])
        elif F1 == 'X3':
            MyFirstFeatureList.append(InitializedInputTestMatrix[:, 2])
        elif F1 == 'X4':
            MyFirstFeatureList.append(InitializedInputTestMatrix[:, 3])
        if F2 == 'X1':
            MySecondFeatureList.append(InitializedInputTestMatrix[:, 0])
        elif F2 == 'X2':
            MySecondFeatureList.append(InitializedInputTestMatrix[:, 1])
        elif F2 == 'X3':
            MySecondFeatureList.append(InitializedInputTestMatrix[:, 2])
        elif F2 == 'X4':
            MySecondFeatureList.append(InitializedInputTestMatrix[:, 3])
        YExact.append(InitializedInputTestMatrix[:, 4])

    return MyFirstFeatureList,MySecondFeatureList,YExact

def DrawLine(Feature1,Feature2,TargetValuesList,BestWeights):
    #Lists To Be plotted
    F1C1List = []
    F2C1List = []
    F1C2List = []
    F2C2List = []
    F1C3List = []
    F2C3List = []
    FirstClassFlag = False
    SecondClassFlag = False
    ThirdClassFlag = False
    # 1 for class 1 and 0 for class 2 and -1 for class 3

    for i in range(60):
        if TargetValuesList[0][i] == 1:
            FirstClassFlag = True
            F1C1List.append(Feature1[i])
            F2C1List.append(Feature2[i])
        elif TargetValuesList[0][i] == 0:
            SecondClassFlag = True
            F1C2List.append(Feature1[i])
            F2C2List.append(Feature2[i])
        elif TargetValuesList[0][i] == -1:
            ThirdClassFlag = True
            F1C3List.append(Feature1[i])
            F2C3List.append(Feature2[i])

    plt.figure('Classification Between Two Classes')
    if ThirdClassFlag == True :
        if FirstClassFlag == True:
            plt.scatter(F1C1List, F2C1List)
            plt.scatter(F1C3List, F2C3List)
        elif SecondClassFlag == True:
            plt.scatter(F1C2List, F2C2List)
            plt.scatter(F1C3List, F2C3List)
    else:
        plt.scatter(F1C1List, F2C1List)
        plt.scatter(F1C2List, F2C2List)


    #  W1*X1 + W2*X2 + Bias
    # we get min and max to scale the line length
    MinimumX = min(Feature1)
    MaximumX = max(Feature1)
    MinimumY=(-(BestWeights[1]*MinimumX)-BestWeights[0])/BestWeights[2]
    MaximumY=(-(BestWeights[1]*MaximumX)-BestWeights[0])/BestWeights[2]
    plt.plot((MinimumX,MaximumX),(MinimumY,MaximumY))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()



def Test(InitializedInputTestMatrix,WeightMatrix,Feature1,Feature2,Class1,Class2,Bias):
    F1List,F2List,TargetListValues = SelectFeatures(0,InitializedInputTestMatrix,Feature1,Feature2,FlagTrainOrTest=1)
    F1Array = np.array(F1List)
    F1Array = F1Array.reshape(40,1)
    F2Array = np.array(F2List)
    F2Array = F2Array.reshape(40,1)
    FinalTestMatrix = GenerateTestInput(F1Array,F2Array,Bias)
    ClassoneMatch = 0
    ClassTwoMatch = 0

    #NewDataSet = {"Class": {"Iris-setosa": 1, "Iris-versicolor": 0, "Iris-virginica": -1}}

    ConfusionMatrix = np.empty([2,2])
    print('Test Matrix Shape = ',FinalTestMatrix.shape)
    print('Weight Matrix Shape = ',WeightMatrix.shape)
    for iteration in range(0, 40):
        #(1,3) * (3,1)
        NetMatrix2 = np.dot(FinalTestMatrix[iteration,:], WeightMatrix)
        pred = Signum(NetMatrix2)
        #Class one Class Two Test

        if (((Class1 == "Iris-setosa") and (Class2 =="Iris-versicolor" )) or((Class1 =="Iris-versicolor" ) and (Class2 == "Iris-setosa"))):
            if ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == 1)):
                ClassoneMatch += 1
            elif ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == 0)):
                ClassTwoMatch += 1

        #Class(1,3)
        elif ((Class1 == "Iris-setosa") and (Class2 =="Iris-virginica" )or((Class1 =="Iris-virginica" ) and (Class2 =="Iris-setosa")) ):
            if ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == 1)):
                ClassoneMatch += 1
            elif ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == -1)):
                ClassTwoMatch += 1

        # Class (2,3)
        elif (((Class1 == "Iris-versicolor") and (Class2 == "Iris-virginica")) or((Class1 == "Iris-virginica") and (Class2 == "Iris-versicolor"))):
            if ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == 0)):
                ClassoneMatch += 1
            elif ((pred == TargetListValues[0][iteration]) and (TargetListValues[0][iteration] == -1)):
                ClassTwoMatch += 1

    ConfusionMatrix[0][0] = ClassoneMatch     #True Positive
    ConfusionMatrix[0][1] = 20 - ClassoneMatch
    ConfusionMatrix[1][0] = 20 - ClassTwoMatch
    ConfusionMatrix[1][1] = ClassTwoMatch     #True Negative

    #how many is positive and we are saying its positive and how many negative and we are saying its negative
    Sum = ConfusionMatrix[0][0] + ConfusionMatrix[1][1]
    Accuracy = (Sum/40) * 100


    return Accuracy, ConfusionMatrix






def AdalineLearningAlgorithm(FinalTrainMatrix,WeightMatrix,Epochs,learningrate,MseThreshold,NumberOfSamples,TargetListValues):
    MSE = 0
    e = 0


    while e < Epochs :

        for iterations in range(NumberOfSamples):
            #1x3 Weights Matrix
            Prediction = np.dot(WeightMatrix.T,FinalTrainMatrix[iterations,:])
            #Compute Error
            Error = TargetListValues[0][iterations] - Prediction
            #Equation = Learningrate * error
            Equation = np.dot(learningrate,Error)
            #Update Weights
            WeightMatrix[0][0] = WeightMatrix[0][0] + np.dot(Equation,FinalTrainMatrix[0][0])
            WeightMatrix[1][0] = WeightMatrix[1][0] + np.dot(Equation,FinalTrainMatrix[0][1])
            WeightMatrix[2][0] = WeightMatrix[2][0] + np.dot(Equation,FinalTrainMatrix[0][2])

        #Mean Square Error Calculation
        for Iterations in range(NumberOfSamples):
            Prediction = np.dot(WeightMatrix.T,FinalTrainMatrix[iterations,:])
            #MSE = (error)^2/2
            MSE += ((TargetListValues[0][Iterations]-Prediction)*(TargetListValues[0][Iterations]-Prediction))/2

        #Average Mean Square Error
        MSE = MSE/NumberOfSamples
        #if Average Mean Square Error < threshold
        if MSE < MseThreshold:
            break
        MSE = 0
        e+=1
    return WeightMatrix



def SingleLayerPerceptronAlgorithm(FinalTrainMatrix,WeightMatrix,learningrate,epochs,NumberOfSamples,TargetValuesList):
    for e in range(epochs):
        for Iteration in range(NumberOfSamples):
            #(1,3) dot product with each row of the (60,3)= (3,) each iteraion
            #We Can Make That Line its similar
            #np.reshape(FinalMatrix,[3,1])
            NetValue = np.dot(WeightMatrix.T, FinalTrainMatrix[Iteration,:])
            Ypred = Signum(NetValue)
            #Apply Equation Of Error
            #W[0] = W[0]+error*learningrate*X
            if (Ypred != TargetValuesList[0][Iteration]):
                #Actual - Predicted
                ErrorValue = TargetValuesList[0][Iteration] - Ypred
                Equation = np.dot(ErrorValue,learningrate)
                WeightMatrix[0][0] = WeightMatrix[0][0] + np.dot(Equation,FinalTrainMatrix[0][0])
                WeightMatrix[1][0] = WeightMatrix[1][0] + np.dot(Equation,FinalTrainMatrix[0][1])
                WeightMatrix[2][0] = WeightMatrix[2][0] + np.dot(Equation,FinalTrainMatrix[0][2])
    return WeightMatrix






def Main(Feature1,Feature2,Class1,Class2,learningRate,epochs,Bias,MSEThreshold):
    #PreProcessing For Training
    FirstThirtyTrainC1,FirstThirtyTrainC2,FirstTwentyTestC1,FirstTwentyTestC2 = ShufflingClasses(Class1,Class2,Bias)
    # Conctenating First thirty of each class (Training Data Matrix) with targets labeled
    InitializedInputMatrix = np.concatenate((FirstThirtyTrainC1,FirstThirtyTrainC2),axis= 0)
    # Conctenating First twenty of each class (Testing Data Matrix) with targets labeled
    InitializedInputTestMatrix = np.concatenate((FirstTwentyTestC1,FirstTwentyTestC2),axis = 0)
    #Selecting Features Entered By User
    F1List,F2List,TargetValuesList = SelectFeatures(InitializedInputMatrix,InitializedInputTestMatrix,Feature1,Feature2,FlagTrainOrTest=0)
    F1Array = np.array(F1List)
    F1Array = F1Array.reshape(60,1)
    F2Array = np.array(F2List)
    F2Array = F2Array.reshape(60,1)
    #Selected Features Matrix
    FinalTrainMatrix = GenerateFinalInputMatrix(F1Array , F2Array , Bias)
    #Weights Matrix
    WeightMatrix = GenerateWeights()

    #SingleLayerPerceptron Will Be Called If The Entry Of The Mse Is Empty
    #UnComment Next Line To Get SingleLayerPerceptron Algorithm
    NewWeights = SingleLayerPerceptronAlgorithm(FinalTrainMatrix, WeightMatrix, learningRate, epochs,len(F1Array),TargetValuesList)
    #NewWeights = AdalineLearningAlgorithm(FinalTrainMatrix,WeightMatrix,epochs,learningRate,MSEThreshold,len(F1Array),TargetValuesList)

    DrawLine(F1Array,F2Array,TargetValuesList,NewWeights)
    Accuracy,ConfusionMatrix = Test(InitializedInputTestMatrix,NewWeights,Feature1,Feature2,Class1,Class2,Bias)
    print('Accuracy = ',Accuracy)
    print('Confusion Matrix = ',ConfusionMatrix)














































