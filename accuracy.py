from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
def marvellousdecision():
   dataset=load_iris()
   data=dataset.data
   target=dataset.target
   data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
   cobj=tree.DecisionTreeClassifier()
   cobj.fit(data_train,target_train)
   output=cobj.predict(data_test)
   accuracy=accuracy_score(target_test,output)
   return accuracy
def main():
    ret=marvellousdecision()
    print("accuracy of decision tree",ret*100,"%")

if __name__=="__main__":
    main()