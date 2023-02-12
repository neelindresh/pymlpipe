from pymlpipe import pipeline
import pandas as pd
from sklearn.datasets import  load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from pymlpipe.tabular import PyMLPipe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import time

ppl=pipeline.PipeLine("IrisData")
mlp=PyMLPipe()
mlp.set_experiment("pipelinecheck")
mlp.set_version(0.1)

def get_data():
    iris_data=load_iris()
    data=iris_data["data"]
    target=iris_data["target"]
    df=pd.DataFrame(data,columns=iris_data["feature_names"])
    #df["target"]=target
    trainx,testx,trainy,testy=train_test_split(df,target)
    
    return {"trainx":trainx,"trainy":trainy,"testx":testx,"testy":testy}

def get_model(model):
    if model==0:
        return LogisticRegression()
    elif model==1:
        return RandomForestClassifier()
    
def train_model(data,model_name):
    with mlp.run():
        trainx,trainy=data["trainx"],data["trainy"]
        mlp.set_tags(["Classification","test run","logisticRegression"])
        model=get_model(model_name)
        model.fit(trainx, trainy)
        
        mlp.scikit_learn.register_model(str(model_name), model)
    
    #print(model)
    #model.fit(trainx, trainy)
    time.sleep(60)
    return model

def evaluate(data,model):
    testx,testy=data["testx"],data["testy"]
    print(model.predict(testx))
    

n1=ppl.add_node("data", get_data,entry_node=True)
for idx,model in enumerate([0,1]):
    ppl.add_node(
        f"model_train{str(idx)}",
        train_model,
        input_nodes=["data"],
        args={"model_name":model},
    ) 
    ppl.add_node(
        f"eval_train{str(idx)}",
        evaluate,
        input_nodes=["data", f"model_train{str(idx)}"],
    )

    #ppl.add_edge(n1, n2)
    #ppl.add_edge(n2, n3)

#n1>>[n2,n3]
ppl.register_dag()
#ppl.run()
    
