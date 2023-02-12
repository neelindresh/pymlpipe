from pymlpipe.tabular import PyMLPipe
import pandas as pd
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import  RandomForestRegressor



training_data=pd.DataFrame([[1,2],[2,4]],columns=["A","B"])
training_data.to_csv("traing.csv")


mlp=PyMLPipe()
mlp.set_experiment("testingMLops1")
mlp.set_version(0.1)


with mlp.run():
    lr=RandomForestRegressor()
    mlp.set_tags(['3','5'])
    mlp.log_param("e", 0.1)
    mlp.log_params({"max_depth":100,"max_iter":500})
    mlp.log_matric("mse",13.10)
    mlp.log_metrics({"rmse":2.5,"mae":.10,'r2':20})
    mlp.register_artifact(artifact=training_data,artifact_name="some.csv",artifact_type='validation')
    mlp.register_artifact_with_path(artifact="traing.csv",)
    
    mlp.scikit_learn.register_model(model_name="linear_regression",model=lr)
    print(mlp.get_info())
    
    
