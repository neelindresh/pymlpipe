

have to open another console to and run this will start processing the pipeline
'''
from pymlpipe import queue

queue.start_server()
'''

then you can start the UI

'''
from pymlpipe.pymlpipeUI import start_ui,start_gui


start_ui(host='0.0.0.0', port=8085,debug=True)

#start_gui()
'''

----
Sample Code snippets for pipeline.

'''
from pymlpipe import pipeline
import pandas as pd
ppl=pipeline.Pipeline("PIPELINEV2")


def node1():
    path="train.csv"
    df=pd.read_csv(path)
    return df
def node2(df):
    stats=df.describe()
    stats.columns=[col+"_node2" for col in stats.columns]
    return stats
def node3(df):
    stats=df.describe()
    stats.columns=[col+"_node3" for col in stats.columns]
    return stats
    
def node4(node1_df,node2_df):
    print(node1_df.append(node2_df))
def node5(node1_df,node2_df,node5_df,node6_df,path):
    print(node1_df.append(node2_df))
    
n1=ppl.add_node("node1", node1,entry_node=True)
n2=ppl.add_node("node2", node3,node_input=["node1"])
n3=ppl.add_node("node3", node2,node_input=["node1"])
n5=ppl.add_node("node5", node2,node_input=["node1"])
n6=ppl.add_node("node6", node2,node_input=["node1"])
n4=ppl.add_node("node4", node4,node_input=["node2","node3"])
n7=ppl.add_node("node7", node5,node_input=["node5","node6"])


ppl.add_edge(n1, n2)
ppl.add_edge(n1, n3)
ppl.add_edge(n1, n5)
ppl.add_edge(n1, n6)

ppl.add_edge(n2, n4)
ppl.add_edge(n3, n4)
ppl.add_edge(n5, n7)
ppl.add_edge(n6, n7)


#n1>>[n2,n3]
ppl.register()
'''

'''
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

ppl=pipeline.Pipeline("IrisData")
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
    n2=ppl.add_node("model_train"+str(idx), train_model,node_input=["data"],args=[model])
    n3=ppl.add_node("eval_train"+str(idx), evaluate,node_input=["data","model_train"+str(idx)])

    ppl.add_edge(n1, n2)
    ppl.add_edge(n2, n3)

#n1>>[n2,n3]
ppl.register()
'''
