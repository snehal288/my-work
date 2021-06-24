from mymodules import *
def calculatedistance(x,y):
    return distance.euclidean(x,y)#((x1,y1) (x2,y2))
class marvellous():#pass parameter as k then put __init_
    def fit(self,trainingdata,trainingtarget):
        self.trainingdata=trainingdata
        self.trainingtarget=trainingtarget
    def predict(self,testdata):
        prediction=[]
        for row in testdata:
            label=self.shortest(row)
            prediction.append(label)
        return prediction
    def shortest(self,row):
        minindex=0
        mindistance=calculatedistance(row,self.trainingdata[0])
        for i in range(1,len(self.trainingdata)):
            distance=calculatedistance(row,self.trainingdata[i])
            if distance <mindistance:
                mindistance=distance
                minindex=i
        return self.trainingtarget[minindex]
def marvellousknn():
    line="_"*50
    iris=load_iris()
    data=iris.data
    target=iris.target
    print(line)
    print("actual data")
    print(line)
    for i in range(len(iris.target)):
        print("ID:%d feature:%s label:%s"%(i,iris.data[i],iris.target[i]))
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
    print(line)
    print("training data set")
    print(line)
    for i in range(len(data_train)):
        print("ID:%d feature:%s label:%s"%(i,data_train[i],target_train[i]))
    print(line)
    print("testing data")
    print(line)
    for i in range(len(data_test)):
        print("ID:%d feature:%s label:%s"%(i,data_test[i],target_test[i]))
    print(line)
    mobj=marvellous() #marvellous(5)
    mobj.fit(data_train,target_train)
    ret=mobj.predict(data_test)
    print("result of ML")
    print(line)
    for i in range(len(data_test)):
        print("ID:%d expectation:%s prediction:%s"%(i,target_test[i],ret[i]))
    print(line)
    icnt=0
    for i in range(len(data_test)):
        if target_test[i]!=ret[i]:
            icnt=icnt+1
    print("number of wrong ans by ml are:",icnt)
    print(line)
    accuracy=accuracy_score(target_test,ret)
    return accuracy
def main():
    ret=marvellousknn()
    print("*"*50)
    print("accuracy of knn is:",ret*100,"%")
    print("*"*50)
if __name__=="__main__":
    main()