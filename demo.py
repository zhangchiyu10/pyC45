import pyC45,csv
if __name__=="__main__":
    #train a C45 decision tree and save the tree as an XML file
    reader = csv.reader(file('./data/training_set.csv'))
    training_obs=[]
    training_cat=[]
    for line in reader:
        training_obs.append(line[:-1])
        training_cat.append(line[-1])
    pyC45.train(training_obs,training_cat,"DecisionTree.xml")
    
    #test the C45 decision tree 
    reader = csv.reader(file('./data/training_set.csv'))
    answer=[]
    testing_obs=[]
    for line in reader:
        testing_obs.append(line[:-1])
        answer.append(line[-1])
    answer.pop(0)
    
    prediction=pyC45.predict("DecisionTree.xml",testing_obs)
    err=0
    for i in range(len(answer)):
        if not answer[i]==prediction[i]:
            err=err+1
    print "error rate=",round(float(err)/len(prediction)*100,2),"%"
    
        