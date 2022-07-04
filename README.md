# PyMLpipe

PyMLpipe is a Python library for MlOps at ease.

* Simple
* Intuative
* Easy to use


## Installation

Use the package manager [pip](https://pypi.org/project/pymlpipe/) to install PyMLpipe.

```bash
pip install pymlpipe
```
or
```bash
pip3 install pymlpipe
```

## Usage

```python
from sklearn.datasets import  load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#import PyMLPipe from tabular 
from pymlpipe.tabular import PyMLPipe


# Initiate the class
mlp=PyMLPipe()
# Set experiment name
mlp.set_experiment("IrisDataV2")
# Set Version name
mlp.set_version(0.2)

iris_data=load_iris()
data=iris_data["data"]
target=iris_data["target"]
df=pd.DataFrame(data,columns=iris_data["feature_names"])
trainx,testx,trainy,testy=train_test_split(df,target)


# to start monitering use mlp.run()
with mlp.run():
    # set tags
    mlp.set_tags(["Classification","test run","logisticRegression"])
    model=LogisticRegression()
    model.fit(trainx, trainy)
    predictions=model.predict(testx)
    # log performace metrics
    mlp.log_matric("Accuracy", accuracy_score(testy,predictions))
    mlp.log_matric("Precision", precision_score(testy,predictions,average='macro'))
    mlp.log_matric("Recall", recall_score(testy,predictions,average='macro'))
    mlp.log_matric("F1", f1_score(testy,predictions,average='macro'))

    # Save train data and test data
    mlp.register_artifact("train", trainx)
    mlp.register_artifact("test", testx,artifact_type="testing")
    # Save the model
    mlp.scikit_learn.register_model("logistic regression", model)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)