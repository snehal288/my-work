import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def Meandata(arr):
    size=len(arr)
    sum=0
    for i in range(size):
        sum=sum+arr[i]
    return(sum/size)
def MarvellousHeadBrain(name):
    dataset=pd.read_csv(name)
    print("size of our data",dataset.shape)
    X=dataset["Head Size(cm^3)"].values
    Y=dataset["Brain Weight(grams)"].values
    print("length of x",len(X))
    print("length of y",len(Y))
    mean_X=Meandata(X) #x bar
    mean_Y=Meandata(Y)
    print("mean of independent variable",mean_X)
    print("mean of dependent variable",mean_Y)
    #m=
    numerator=0
    denomenator=0
    for i in range(len(X)):
        numerator=numerator+(X[i]-mean_X)*(Y[i]-mean_Y)
        denomenator=denomenator+(X[i]-mean_X)**2
    m=numerator/denomenator
    print("value of m",m)
    #y=mx+c
    #c=y-mx
    c=mean_Y-(m*mean_X)
    print("value of y intercept is",c)
    #graph
    X_Start=np.min(X)-200
    X_End=np.max(X)+200
    x=np.linspace(X_Start,X_End)
    y=m*x+c
    plt.plot(x,y,color='r',label="regression line")
    plt.scatter(X,Y,color='b',label="data plot")
    plt.xlabel("head size")
    plt.ylabel("brain weight")
    plt.legend()
    plt.show()
    #r square

def main():
    #print("enter file name")
    #name=input()
    MarvellousHeadBrain("MarvellousHeadBrain.csv")
if __name__=="__main__":
    main()