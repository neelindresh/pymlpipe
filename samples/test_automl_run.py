from pymlpipe.automl import AutoMLPipe
from sklearn.datasets import  load_iris
import pandas as pd
import numpy as np

def main():
    
    iris_data=load_iris()
    data=iris_data["data"]
    target=iris_data["target"]

    df=pd.DataFrame(data,columns=iris_data["feature_names"])
    automl_obj=AutoMLPipe("IrisAutoML","classification",
                            "precision",
                            df,
                            target,
                            tags=["new_data","clf"],
                            transform=True,
                            scale='normalize',
                            register_model=True,
                            version=1.0,
                            exclude=['log_reg',"lgbmc"])
    preds,result=automl_obj.run_automl(tune=False,tune_best=False)
    
    #DataFrame with comparative metrics of all the models
    print(result)
    #Dictionary with model names and the predictions 
    print(preds)

if __name__ == '__main__':
    main()

