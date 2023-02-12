from sklearn.datasets import  load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from pymlpipe.tabular import PyMLPipe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



mlp=PyMLPipe()
mlp.set_experiment("IrisDataV2")
mlp.set_version(0.1)

iris_data=load_iris()
data=iris_data["data"]
target=iris_data["target"]
df=pd.DataFrame(data,columns=iris_data["feature_names"])
#df["target"]=target
trainx,testx,trainy,testy=train_test_split(df,target)

with mlp.run():
    mlp.set_tags(["Classification","test run","logisticRegression"])
    model=LogisticRegression()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    mlp.log_metrics({"Accuracy":accuracy_score(testy,predictions),
                     "Precision": precision_score(testy,predictions,average='macro'),
                     "Recall": recall_score(testy,predictions,average='macro'),
                     "F1": f1_score(testy,predictions,average='macro')
                     })
    mlp.register_artifact("train.csv", trainx)
    mlp.register_artifact("test.csv", testx,artifact_type="testing")
    mlp.scikit_learn.register_model("logistic regression", model)
    mlp.explainer(model,trainx)
    
    
    



with mlp.run():
    mlp.set_tags(["Classification","test run","dtree"])
    model=DecisionTreeClassifier()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    
    mlp.log_metrics({"Accuracy":accuracy_score(testy,predictions),"Precision": precision_score(testy,predictions,average='macro')})
    
    mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
    mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
    
    #mlp.log_metrics({"r2":0.1,"mse":1.1})
    mlp.register_artifact("train.csv", trainx)
    mlp.register_artifact("test.csv", testx,artifact_type="testing")
    mlp.scikit_learn.register_model("dtree", model)
    mlp.explainer(model,trainx)

with mlp.run():
    mlp.set_tags(["Classification","test run","rf"])
    model=RandomForestClassifier()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    
    mlp.log_metric("Accuracy", accuracy_score(testy,predictions))
    mlp.log_metric("Precision", precision_score(testy,predictions,average='macro'))
    mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
    mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
    mlp.register_artifact("train.csv", trainx,)
    mlp.register_artifact("test.csv", testx,artifact_type="testing")
    mlp.scikit_learn.register_model("randomForest", model)
    mlp.explainer(model,trainx)

with mlp.run():
    mlp.set_tags(["Classification","test run","xgb"])
    model=XGBClassifier()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    
    mlp.log_metric("Accuracy", accuracy_score(testy,predictions))
    mlp.log_metric("Precision", precision_score(testy,predictions,average='macro'))
    mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
    mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
    mlp.register_artifact("train.csv", trainx)
    mlp.register_artifact("test.csv", testx,artifact_type="testing")
    mlp.scikit_learn.register_model("xgboost", model)
    mlp.explainer(model,trainx)   

with mlp.run():
    mlp.set_tags(["Classification","test run","xgb"])
    model=AdaBoostClassifier()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    
    mlp.log_metric("Accuracy", accuracy_score(testy,predictions))
    mlp.log_metric("Precision", precision_score(testy,predictions,average='macro'))
    mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
    mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
    mlp.register_artifact("train.csv", trainx)
    mlp.register_artifact("test.csv", testx,artifact_type="testing")
    mlp.scikit_learn.register_model("adaboost", model)
    mlp.explainer(model,trainx)   
    
