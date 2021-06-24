from sklearn import tree


def main():
    #step1&2
    Features = [[35, "Rough"], [47, "Rough"], [90, "Smooth"], [48, "Rough"], [90, "Smooth"], [35, "Rough"],
                [92, "Smooth"], [35, "Rough"], [35, "Rough"], [35, "Rough"], [96, "Smooth"], [43, "Rough"],
                [110, "Smooth"], [35, "Rough"], [95, "Smooth"]]

    Label= ["Tennis", "Tennis", "Cricket", "Tennis", "Cricket", "Tennis", "Cricket", "Tennis", "Tennis", "Tennis",
              "Cricket", "Tennis", "Cricket", "Tennis", "Cricket"]
    #step 3
    dobj=tree.DecisionTreeClassifier()
    #step4
    dobj.fit(Features,Label)  #training
    #STEP5
    result=dobj.predict([[40,1]]) #testing
    print("ball is",result)


if __name__ == "__main__":
    main()
