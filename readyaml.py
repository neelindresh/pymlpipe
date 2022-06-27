'''
import yaml

with open("/Users/indreshbhattacharya/Desktop/WorkSpace/mlflowbyme/modelrun/tetsingML/neel/info.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    fruits_list = yaml.load(file, Loader=yaml.FullLoader)

    print(fruits_list)
    
    




pickled_model = pickle.load(open('/Users/indreshbhattacharya/Desktop/WorkSpace/mlflowbyme/modelrun/testingMLops/neel/models/linear_regression.pkl', 'rb'))

print(pickled_model)


'''
'''
import pickle
import dill
import pandas as pd
import os
training_data=pd.DataFrame([[1,2],[2,4]],columns=["A","B"])

def main_abc(training_data):
    tt=training_data.sum(axis=1)
    print(os.getcwd())
    return tt

def func(a=None):
    if a == None:
        return 10
    else:
        return a
main_abc(training_data)
dill.dump(main_abc, open('main.pkl', 'wb'))
'''
import pickle
import dill
import pandas as pd
training_data=pd.DataFrame([[1,2],[2,4]],columns=["A","B"])

pickled_model = dill.load(open('main.pkl', 'rb'))
print(pickled_model(training_data))
