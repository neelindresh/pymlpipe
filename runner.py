from pymlpipe import PyMLPipe
import pandas as pd
from sklearn.linear_model import  LinearRegression



training_data=pd.DataFrame([[1,2],[2,4]],columns=["A","B"])
training_data.to_csv("traing.csv")


mlp=PyMLPipe()
mlp.set_experiment("testingMLops1")
mlp.set_version(0.1)


with mlp.run():
    lr=LinearRegression()
    mlp.set_tags(['1','2'])
    mlp.log_param("e", 0.1)
    mlp.log_params({"max_depth":100,"max_iter":500})
    mlp.log_matric("mse",12.10)
    mlp.log_metrics({"rmse":6.5,"mae":1.10,"r2":1.11})
    mlp.register_artifact(artifact=training_data,artifact_name="some.csv")
    mlp.register_artifact_with_path(artifact="traing.csv")
    print(mlp.get_info())
    mlp.scikit_learn.register_model(model_name="linear_regression",model=lr)
    
    
