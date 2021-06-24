from sklearn import tree
#rough 1 smooth 0
#tennis 1 cricket 2
def main():
    #step1&2
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [35,1],
                [92, 0], [35, 1], [35, 1], [35, 1], [96,0], [43,1],
                [110, 0], [35, 1], [95, 0]]

    Label= [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2,1,2]
    #step 3
    dobj=tree.DecisionTreeClassifier()
    #step4
    dobj.fit(Features,Label)  #training
    #STEP5
    result=dobj.predict([[40,1]]) #testing
    print("ball is",result)


if __name__ == "__main__":
    main()
