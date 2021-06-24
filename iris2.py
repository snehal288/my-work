from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
def main():#loading data set
    dataset=load_iris()
    print("feature of dataset")
    print(dataset.feature_names)
    print("target.names of dataset")
    print(dataset.target_names) #target label
    index=[1,2,3,4,5,6,7,8,9,10,51,52,53,54,55,56,57,58,59,60,101,102,103,104,105,106,107,108,109]
    test_target=dataset.target[index]
    test_features=dataset.data[index]
    train_target=np.delete(dataset.target,index)
    train_features=np.delete(dataset.data,index,axis=0)
    obj=tree.DecisionTreeClassifier()
    obj.fit(train_features,train_target)
    result=obj.predict(test_features)
    print("result prediction by ML",result)
    print("result expected",test_target)

if __name__=="__main__":
    main()