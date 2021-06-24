from sklearn import tree
#rough 1 smooth 0
#tennis 1 cricket 2
def MarvellousML(weight,surface):
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
    result=dobj.predict([[weight,surface]]) #testing
    if result==1:
        print("your object is tennis ball")
    else:
        print("your object looks like cricket ball")
    print("ball is",result)
def main():
    print("********************supervised machine learning*************")
    print("enter weight of object")
    weight=int(input())
    print("enter surface type of object")
    surface=input()
    if surface.lower()=="rough":#ROUGH,Rough convert kel lower madhe
        surface=1
    elif surface.lower()=="smooth":
        surface=0
    else:
        print("invalid input")
        return
    MarvellousML(weight,surface)

if __name__ == "__main__":
    main()
