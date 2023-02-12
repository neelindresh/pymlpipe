import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from pymlpipe.tabular import PyMLPipe
df=pd.read_csv("train.csv")
encoders=["area_code","state","international_plan","voice_mail_plan","churn"]

for i in encoders:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i])
    
    
trainy=df["churn"]
trainx=df[['state', 'account_length', 'area_code', 'international_plan',
       'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_day_charge', 'total_eve_minutes',
       'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
       'total_night_calls', 'total_night_charge', 'total_intl_minutes',
       'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']]


class Model(torch.nn.Module):
    def __init__(self,col_size):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Linear(col_size,15),
            torch.nn.ReLU(),
            torch.nn.Linear(15,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
        )
        '''
        self.linear_layer_1=
        self.relu_1=torch.nn.ReLU()
        self.linear_layer_2=torch.nn.Linear(15,10)
        self.relu_2=torch.nn.ReLU()
        self.linear_layer_3=torch.nn.Linear(10,1)
        self.linear_layer_4=torch.nn.Linear(10,1)
        '''
        
        
    def forward(self,x):
        out=self.seq(x)
        
        return torch.sigmoid(out)
        
model=Model(len(trainx.columns))

train_x,test_x,train_y,test_y=train_test_split(trainx,trainy)

train_x=torch.from_numpy(train_x.values)
train_x=train_x.type(torch.FloatTensor)
train_y=torch.from_numpy(train_y.values)
train_y=train_y.type(torch.FloatTensor)

test_x=torch.from_numpy(test_x.values)
test_x=test_x.type(torch.FloatTensor)
test_y=torch.from_numpy(test_y.values)
test_y=test_y.type(torch.FloatTensor)


optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

criterion=torch.nn.BCELoss()


def validate(model,testx,testy):
    prediction=model(testx)
    prediction=torch.where(prediction>.5,1,0)
    accu=accuracy_score(prediction.detach().numpy(),test_y.unsqueeze(1).detach().numpy())
    f1=f1_score(prediction.detach().numpy(),test_y.unsqueeze(1).detach().numpy())
    return {"accuracy":accu,"f1":f1}


epochs=100
batch_size=1000

mlp=PyMLPipe()
mlp.set_experiment("Pytorch")
mlp.set_version(0.2)

with mlp.run():
    mlp.register_artifact("churndata.csv",df)
    mlp.log_params({
        "lr":0.01,
        "optimizer":"SGD",
        "loss_fuction":"BCEloss"
    })
    for epoch in range(epochs):
        loss_batch=0
        for batch in range(1000,5000,1000):
            optimizer.zero_grad()
            train_data=train_x[batch-1000:batch]
            output=model(train_data)
            loss=criterion(output,train_y[batch-1000:batch].unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss_batch+=loss.item()

        metrics=validate(model,test_x,test_y)
        metrics["loss"]=loss_batch
        metrics["epoch"]=epoch
        mlp.log_metrics_continious(metrics)
    mlp.pytorch.register_model("pytorch_example1", model)
        