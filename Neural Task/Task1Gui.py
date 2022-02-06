import numpy as np
import matplotlib.pyplot as plt
from tkinter import*
from tkinter import messagebox
import pandas as pd
from Task1NeuralNetwork import *


MainWindow = Tk()
MainWindow.geometry('500x500')
MainWindow.title('Main GUI')
Data = pd.read_csv('IrisData.txt',',',names = ['X1','X2','X3','X4','Class'])
Data = Data.drop(Data.index[0])

#Functions X(number,class)
#All Possible Combinations
def Draw(FeatureSelected1,FeatureSelected2):
    if (FeatureSelected1 == 'X1') and (FeatureSelected2 == 'X2'):
        X = pd.to_numeric(Data['X1'])
        Y = pd.to_numeric(Data['X2'])
        X11 = list(X[0:50])
        X21 = list(Y[0:50])
        X12 = list(X[50:100])
        X22 = list(Y[50:100])
        X13 = list(X[100:150])
        X23 = list(Y[100:150])
        plt.figure('X1 And X2 Samples')
        plt.scatter(X11, X21)
        plt.scatter(X12, X22)
        plt.scatter(X13, X23)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    elif (FeatureSelected1 == 'X1') and (FeatureSelected2 == 'X3'):
        X = pd.to_numeric(Data['X1'])
        Y = pd.to_numeric(Data['X3'])
        X11 = list(X[0:50])
        X31 = list(Y[0:50])
        X12 = list(X[50:100])
        X32 = list(Y[50:100])
        X13 = list(X[100:150])
        X33 = list(Y[100:150])
        plt.figure('X1 And X3 Samples')
        plt.scatter(X11, X31)
        plt.scatter(X12, X32)
        plt.scatter(X13, X33)
        plt.xlabel('X1')
        plt.ylabel('X3')
        plt.show()
    elif (FeatureSelected1 == 'X1') and (FeatureSelected2 == 'X4'):
        X = pd.to_numeric(Data['X1'])
        Y = pd.to_numeric(Data['X4'])
        X11 = list(X[0:50])
        X41 = list(Y[0:50])
        X12 = list(X[50:100])
        X42 = list(Y[50:100])
        X13 = list(X[100:150])
        X43 = list(Y[100:150])
        plt.figure('X1 And X4 Samples')
        plt.scatter(X11, X41)
        plt.scatter(X12, X42)
        plt.scatter(X13, X43)
        plt.xlabel('X1')
        plt.ylabel('X4')
        plt.show()
    elif (FeatureSelected1 == 'X2') and (FeatureSelected2 == 'X3'):
        X = pd.to_numeric(Data['X2'])
        Y = pd.to_numeric(Data['X3'])
        X21 = list(X[0:50])
        X31 = list(Y[0:50])
        X22 = list(X[50:100])
        X32 = list(Y[50:100])
        X23 = list(X[100:150])
        X33 = list(Y[100:150])
        plt.figure('X2 And X3 Samples')
        plt.scatter(X21, X31)
        plt.scatter(X22, X32)
        plt.scatter(X23, X33)
        plt.xlabel('X2')
        plt.ylabel('X3')
        plt.show()
    elif (FeatureSelected1 == 'X2') and (FeatureSelected2 == 'X4'):
        X = pd.to_numeric(Data['X2'])
        Y = pd.to_numeric(Data['X4'])
        X21 = list(X[0:50])
        X41 = list(Y[0:50])
        X22 = list(X[50:100])
        X42 = list(Y[50:100])
        X23 = list(X[100:150])
        X43 = list(Y[100:150])
        plt.figure('X2 And X4 Samples')
        plt.scatter(X21, X41)
        plt.scatter(X22, X42)
        plt.scatter(X23, X43)
        plt.xlabel('X2')
        plt.ylabel('X4')
        plt.show()
    elif (FeatureSelected1 == 'X3') and (FeatureSelected2 == 'X4'):
        X = pd.to_numeric(Data['X3'])
        Y = pd.to_numeric(Data['X4'])
        X31 = list(X[0:50])
        X41 = list(Y[0:50])
        X32 = list(X[50:100])
        X42 = list(Y[50:100])
        X33 = list(X[100:150])
        X43 = list(Y[100:150])
        plt.figure('X3 And X4 Samples')
        plt.scatter(X31, X41)
        plt.scatter(X32, X42)
        plt.scatter(X33, X43)
        plt.xlabel('X3')
        plt.ylabel('X4')
        plt.show()

    return

def Run(Feature1,Feature2,Class1,Class2,LearningRate,Epochs,Bias,MSEThreshold):
    Main(Feature1,Feature2,Class1,Class2,LearningRate,Epochs,Bias,MSEThreshold)


#Labels
Feature1 = Label(MainWindow,text = 'Feature 1',bg = 'red',fg = 'black').grid(row = 0,column = 0,padx = 10,pady = 10)
Feature2 = Label(MainWindow,text = 'Feature 2',bg = 'red',fg = 'black').grid(row = 1,column = 0,padx = 10,pady = 10)
Class1 = Label(MainWindow,text = 'Class 1',bg = 'red',fg = 'black').grid(row = 0,column = 5,padx = 10,pady = 10)
Class2 = Label(MainWindow,text = 'Class 2',bg = 'red',fg = 'black').grid(row = 1,column = 5,padx = 10,pady = 10)
LearningRateLabel = Label(MainWindow,text = 'Learning Rate',bg = 'red',fg = 'black').grid(row = 2,column = 0,padx = 10,pady = 10)
Epochs = Label(MainWindow,text = 'Epochs',bg = 'red',fg = 'black').grid(row = 2,column = 3,padx = 10,pady = 10)
Bias = Label(MainWindow,text = 'Bias',bg = 'red',fg = 'black').grid(row = 2,column = 5,padx = 10,pady = 10)
MseThreshold = Label(MainWindow,text ='Mse Threshold',bg = 'red',fg = 'black').grid(row = 3,column = 0,padx = 10,pady = 10)
FeaturesList = ['X1','X2','X3','X4']
ClassesList = ['Iris-setosa','Iris-versicolor','Iris-virginica']
BiasList = ['0','1']

#Default Values Of The Features List
FeatureSelected1 = StringVar()
FeatureSelected1.set(FeaturesList[0])
FeatureSelected2 = StringVar()
FeatureSelected2.set(FeaturesList[0])

#Default Values Of The Classes List
ClassSelected1 = StringVar()
ClassSelected1.set(ClassesList[0])
ClassSelected2 = StringVar()
ClassSelected2.set(ClassesList[0])
BiasSelected = StringVar()
BiasSelected.set(BiasList[0])

#Menus
FeaturesMenu1 = OptionMenu(MainWindow, FeatureSelected1 ,  *FeaturesList ).grid(row = 0,column = 1)
FeaturesMenu2 = OptionMenu(MainWindow, FeatureSelected2 ,  *FeaturesList ).grid(row = 1,column = 1)
ClassesMenu1 = OptionMenu(MainWindow, ClassSelected1 ,  *ClassesList ).grid(row = 0,column = 6,padx = 10,pady = 10)
ClassesMenu2 = OptionMenu(MainWindow, ClassSelected2 ,  *ClassesList ).grid(row = 1,column = 6,padx = 10,pady = 10)
BiasMenu = OptionMenu(MainWindow, BiasSelected ,  *BiasList ).grid(row = 2,column = 6,padx = 10,pady = 10)


#Entries

LearningRateEntry = Entry(MainWindow,width = 5)
LearningRateEntry.grid(row = 2,column = 1)
EpochsEntry = Entry(MainWindow,width = 5)
EpochsEntry.grid(row = 2,column = 4)
MSEThresholdEntry = Entry(MainWindow,width = 5)
MSEThresholdEntry.grid(row = 3,column = 1)
#Buttons

RunButton = Button(MainWindow,text = 'Run',width = 10,command = lambda : Run(FeatureSelected1.get(),FeatureSelected2.get(),ClassSelected1.get(),ClassSelected2.get(),float(LearningRateEntry.get()),int(EpochsEntry.get()),int(BiasSelected.get()),float(MSEThresholdEntry.get())))
RunButton.grid(row = 5,column = 5)
DrawButton = Button(MainWindow,text = 'Draw Data',width = 10,command = lambda : Draw(FeatureSelected1.get(),FeatureSelected2.get()))
DrawButton.grid(row = 5,column = 6)


MainWindow.mainloop()


