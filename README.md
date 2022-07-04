
![alt text](https://github.com/neelindresh/pymlpipe/blob/main/static/logo.svg?raw=true)
# PyMLpipe

PyMLpipe is a Python library for ease Machine Learning Model monitoring and Deployment.

* Simple
* Intuative
* Easy to use

Please Find the Full [documentation](https://neelindresh.github.io/pymlpipe.documentation.io/) here!

## Installation

Use the package manager [pip](https://pypi.org/project/pymlpipe/) to install PyMLpipe.

```bash
pip install pymlpipe
```
or
```bash
pip3 install pymlpipe
```


## Tutorial 

* Load the python package

```python
from pymlpipe.tabular import PyMLPipe
```

* Initiate the `PyMLPipe` class

```python
mlp=PyMLPipe()
```

* Set an Experiment Name `[Optional]`-Default experiment name is `'0'`

```python
mlp.set_experiment("IrisDataV2")
```

* Set a version `[Optional]`-Default there is no version

```python
mlp.set_version(0.1)
```

* Initiate the context manager - This is create a unique ID for each model run. 
    -  when `.run()` is used - Automatic  unique ID is generated 
    - you can also provide `runid` argument in the `.run()` this will the use the given `runid` for next storing.

```python
with mlp.run():
```
Or

```python
with mlp.run(runid='mlopstest'):
```

*  Set a Tag `[Optional]` by using `set_tag()`-Default there is no tags

```python
 mlp.set_tag('tag')
```
Or 

*  Set multiple Tags `[Optional]` by using `set_tags()`-Default there is no tags
```python
mlp.set_tags(["Classification","test run","logisticRegression"])

```

*  Set Metrics values `[Optional]` by using `log_matric(metric_name,metric_value)`-Default there is no metrics
This will help in comparing performance of different models and model versions
```python
mlp.log_matric("Accuracy", accuracy_score(testy,predictions))


mlp.log_matric("Accuracy", .92)

```

*  Set multiple Metrics values `[Optional]` by using `log_matrics({metric_name:metric_value})`-Default there is no metrics
```python
mlp.log_matrics(
    {
        "Accuracy": accuracy_score(testy,predictions),
        "Precision": precision_score(testy,predictions,average='macro'),
        "Recall", recall_score(testy,predictions,average='macro'),
    }
)


mlp.log_matrics(
    {
        "Accuracy": .92,
        "Precision": .87,
        "Recall", .98,
    }
)


```
   
*  Save an artifact `[Optional]` - You can save training/testing/validation/dev/prod data for monitoring and comparison
    - This will also help in generating `DATA SCHEMA`
    - `register_artifact()` -takes 3 arguments 
        - name of artifact
        - Pandas Dataframe
        - type of artifact - `[training, testing, validation, dev, prod]`
    - You can also use `register_artifact_with_path()` - This will save the artifact from the disk. 
        - Path for the file
        - type of artifact - `[training, testing, validation, dev, prod]`

```python
    mlp.register_artifact("train.csv", trainx)

    mlp.register_artifact("train.csv", trainx)
```
* Register Model `[Optional]` - You can register the model. This will help in Quick deployment

```python
   mlp.scikit_learn.register_model("logistic regression", model)
```

## Quick Start

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

## Launch UI

To start the UI 

```bash
pymlpipeui 
```
or 
```python
from pymlpipe.pymlpipeUI import start_ui


start_ui(host='0.0.0.0', port=8085)
```
#### Sample UI


![alt text](https://github.com/neelindresh/pymlpipe/blob/development/static/Screenshot%202022-07-04%20at%201.42.35%20PM.png?raw=true)

---

![alt text](https://github.com/neelindresh/pymlpipe/blob/development/static/Screenshot%202022-07-04%20at%201.42.52%20PM.png?raw=true)

---
#### One Click Deployment -click the deploy button to deploy the model and get a endpoint


![alt text](https://github.com/neelindresh/pymlpipe/blob/development/static/Screenshot%202022-07-04%20at%201.43.03%20PM.png?raw=true)

---


![alt text](https://github.com/neelindresh/pymlpipe/blob/development/static/Screenshot%202022-07-04%20at%201.43.52%20PM.png?raw=true)

---

## Send the data to the Prediction end point in the format

 - Each list is a row of data
```python
{
        "data":[
            [
                5.6,
                3.0,
                4.5,
                1.5
            ],
            [
                5.6,
                3.0,
                4.5,
                1.5
            ]
        ]
    }
```

![alt text](https://github.com/neelindresh/pymlpipe/blob/development/static/Screenshot%202022-07-04%20at%201.44.05%20PM.png?raw=true)

---


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)