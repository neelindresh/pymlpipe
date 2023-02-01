import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,BaggingRegressor,AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier,LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, HuberRegressor, PoissonRegressor,PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import LinearSVC,SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score,mean_absolute_error,mean_squared_error,make_scorer
from pymlpipe.tabular import PyMLPipe 
#from tabular import PyMLPipe
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from itertools import chain


class AutoMLPipe():
    def __init__(self,exp_name,task,metric,data,label,tags=[],test_size=0.20,version=1.0,transform=False,scale='standard',cols_to_scale=[],categorical_cols=[],register_model=False,register_artifacts=False,exclude=[]):
        '''
        exp_name: name of experiment
        task: regression/classification
        metric: for classification -> accuracy,recall,precision,f1/ for regression -> MAE,MSE,RMSE,R2 Score
        data: data on which the model to be fit
        label: target variable
        tags: list of custom-tags for the run
        test_size: size of test dataset
        version: experiment version
        transform:bool
        scale: 'standard'/'minmax'/'normalize'
        cols_to_scale: list of columns to scale. Should be numeric or float
        register_model: register experiement model
        register_artifacts: register experiment artifacts
        exclude: models to be excluded during autoML runs
        '''
        self.exp_name=exp_name
        self.task=task
        self.metric=metric
        self.data=data
        self.label=label
        self.test_size=test_size
        self.version=version
        self.exclude=exclude
        self.transform=transform
        self.scale=scale
        self.cols_to_scale=cols_to_scale
        self.categorical_cols=categorical_cols
        self.mlp=PyMLPipe()
        self.register_model=register_model
        self.register=register_artifacts
        self.tags=tags
        self.classification_models={
                'log_reg': LogisticRegression(),
                'adac':AdaBoostClassifier(),
                'bagc': BaggingClassifier(),
                'etc' : ExtraTreesClassifier(),
                'gbc' : GradientBoostingClassifier(),
                'rfc': RandomForestClassifier(),
                'dtc': DecisionTreeClassifier(),
                'rc': RidgeClassifier(),
                'sgdc':SGDClassifier(),
                'pac':PassiveAggressiveClassifier(),
                'svc': LinearSVC(),
                'mlpc': MLPClassifier(),
                'xgbc': XGBClassifier(n_jobs=-1),
                'lgbmc': LGBMClassifier(n_jobs=-1),
                'cbc': CatBoostClassifier()}
        self.regression_models={
            'lr': LinearRegression(),
            'svr' : SVR(),
            'adar' : AdaBoostRegressor(),
            'dtr' : DecisionTreeRegressor(),
            'lasso' : Lasso(),
            'ridge' : Ridge(),
            'mlpr' : MLPRegressor(),
            'rfr' : RandomForestRegressor(),
            'etr' : ExtraTreesRegressor(),
            'gbr' : GradientBoostingRegressor(),
            'bagr' : BaggingRegressor(),
            'enet' : ElasticNet(),
            'par': PassiveAggressiveRegressor(),
            'bay' : BayesianRidge(),
            'hubr' : HuberRegressor(),
            'poi' : PoissonRegressor(),
            'xgbr': XGBRegressor(n_jobs=-1),
            'lgbmr': LGBMRegressor(n_jobs=-1),
            'cbr': CatBoostRegressor()

        }
        self.param_grid=dict()
        self.param_grid['log_reg']={
            'penalty': ['l1','l2'],
            'C': [0.1,1],
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],
        }
        self.param_grid['pac']={
            'C': [0.01,0.1,1],

        }
        self.param_grid['par']={
            'C': [0.1,0.5,1],

        }
        self.param_grid['rc']={
            'alpha':[0.01,0.1,1],
            'solver': ['auto','sag','cholesky']
        }
        self.param_grid['sgdc']={
            'loss': ['hinge','log_loss', 'modified_huber','squared_error'],
            'penalty': ['l1','l2']
        }
        self.param_grid['dtc']={
            'criterion': ['gini','entropy'],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['adac']={
            'n_estimators': [10,100,500],
            'learning_rate': [0.01,0.1,1],
        }
        self.param_grid['bagc']={
            'n_estimators': [10,100,500],
        }
        self.param_grid['bagr']={
            'n_estimators': [10,100,500],
        }
        self.param_grid['etc']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['etr']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['gbc']={
            'n_estimators': [10,100,500],
            'learning_rate': [0.01,0.1],
            'criterion': ['friedman_mse','squared_error']
        }
        self.param_grid['gbr']={
            'n_estimators': [10,100,500],
            'learning_rate': [0.01,0.1],
            'criterion': ['friedman_mse','squared_error']
        }
        self.param_grid['rfc']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['rfr']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['svc']={
            'loss': ['hinge','log_loss', 'modified_huber','squared_error'],
            'C': [0.1,0.5,1]
        }
        self.param_grid['mlpc']={
            'activation': ['tanh','relu'],
            'solver': ['sgd','adam']
        }
        self.param_grid['mlpr']={
            'activation': ['tanh','relu'],
            'solver': ['sgd','adam']
        }
        self.param_grid['lr']={
                'n_jobs' : [-1]
            }
        self.param_grid['svr']={
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
            ,'gamma' : ['scale','auto']
            , 'C': [0.1,0.5,1]
        }
        self.param_grid['adar']={
            'n_estimators': [10,100,500],
            'learning_rate': [0.01,0.1,1],
            'loss' : ['linear','square','exponential']
        }
        self.param_grid['dtr']={
            #'criterion': ['gini','entropy'],
            'splitter' : ['best'],
            'max_depth': [None,2,3],
            'min_samples_split': [2,3,4]
        }
        self.param_grid['lasso']={
            'selection' : ['cyclic', 'random']
        }
        self.param_grid['ridge']={
            'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }
        self.param_grid['poi']={
            'alpha': [0.5,1,1.5]
        }
        self.param_grid['hubr']={
            'epsilon': [1.35,1.5,1.75,2]
        }
        self.param_grid['enet']={
            'l1_ratio': [0.3,0.5,0.6,0.7]
        }
        self.param_grid['bay']={
            'n_iter': [100,300,500]
        }
        self.param_grid['xgbc']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'learning_rate': [0.01,0.1]
        }
        self.param_grid['lgbmc']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'num_leaves': [20,30,40]
        }
        self.param_grid['cbc']={
            'n_estimators': [10,100,500],
            'max_depth': [2,3],
           'learning_rate': [0.01,0.1]
        }
        self.param_grid['xgbr']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'learning_rate': [0.01,0.1]
        }
        self.param_grid['lgbmr']={
            'n_estimators': [10,100,500],
            'max_depth': [None,2,3],
            'num_leaves': [20,30,40]
        }
        self.param_grid['cbr']={
            'n_estimators': [10,100,500],
            'max_depth': [2,3],
           'learning_rate': [0.01,0.1]
        }   

    def run_automl(self,tune=False,tune_best=False):
        '''
        tune: param tune all the models
        tune_best: param tune the best model
        '''

        # Set Experiment name
        self.mlp.set_experiment(self.exp_name)
        # Set Version name
        self.mlp.set_version(self.version)

        if self.transform==True:
            numeric_cols=self.data.select_dtypes(include=[np.number]).columns
            numeric_cols=[item for item in numeric_cols if item not in self.categorical_cols]
            cols_to_scale=self.cols_to_scale
            if cols_to_scale==[]:
                cols_to_scale=numeric_cols
            check =  all(item in numeric_cols for item in cols_to_scale)
            if check==True:
                if self.scale=='standard':
                    scaler = StandardScaler()
                elif self.scale=='minmax':
                    scaler = MinMaxScaler()
                elif self.scale=='normalize':
                    scaler = Normalizer()
                elif self.scale=='robust':
                    scaler = RobustScaler()                       
                scaler.fit(self.data[cols_to_scale])
                self.data[cols_to_scale] = scaler.transform(self.data[cols_to_scale])     
            else:
                print('Scaling operation cannot be completed as column type is not int/float')

        if self.categorical_cols!=[]:
            self.data = pd.get_dummies(self.data, columns = self.categorical_cols)

        trainx,testx,trainy,testy=train_test_split(self.data,self.label,test_size=self.test_size)      
        result=pd.DataFrame()
        prediction_set={}
        if self.task=='classification':
            for model_name,model in tqdm(self.classification_models.items()):
                if model_name not in self.exclude:        
                    if tune==True:
                        try:   
                            predictions,result_set=self.param_tune_model(model_name,trainx,testx,trainy,testy)
                            predictions=predictions.tolist()
                            if model_name=='cbc':
                                predictions=list(chain(*predictions))
                            fin=dict()
                            fin['name']=model_name
                            fin['accuracy']=result_set['accuracy']
                            fin['precision']=result_set['precision']
                            fin['recall']=result_set['recall']
                            fin['f1']=result_set['f1_score']
                        except Exception as e:
                            print (e)
                            continue
                    
                    else:
                        try:
                            with self.mlp.run():
                                default_tags=[model_name,"Classification"]
                                tag_list=default_tags+self.tags
                                self.mlp.set_tags(tag_list)
                            
                                model=model
                                model.fit(trainx, trainy)
                                predictions=model.predict(testx)
                                predictions=predictions.tolist()
                                if model_name=='cbc':
                                    predictions=list(chain(*predictions))

                                self.mlp.log_metric("accuracy", accuracy_score(testy,predictions))
                                self.mlp.log_metric("precision", precision_score(testy,predictions,average='macro'))
                                self.mlp.log_metric("recall", recall_score(testy,predictions,average='macro'))
                                self.mlp.log_metric("f1_score", f1_score(testy,predictions,average='macro'))

                                if self.register==True:
                                    self.mlp.register_artifact("train", trainx)
                                    self.mlp.register_artifact("test", testx,artifact_type="testing")
                                if self.register_model==True:
                                    self.mlp.scikit_learn.register_model(model_name, model)
                            
                                result1=self.mlp.get_info()
                                fin=dict()
                                fin['name']=model_name
                                fin['accuracy']=result1['metrics']['accuracy']
                                fin['precision']=result1['metrics']['precision']
                                fin['recall']=result1['metrics']['recall']
                                fin['f1']=result1['metrics']['f1_score']
                        except Exception as e:
                            print(e)                        
                            continue
                    
                    prediction_set[model_name]=predictions
                    result=result.append(fin,ignore_index=True)
        elif self.task=='regression':
            for model_name,model in tqdm(self.regression_models.items()):
                if model_name not in self.exclude:
                    if tune==True:
                        try:
                            predictions,result_set=self.param_tune_model(model_name,trainx,testx,trainy,testy)
                            predictions=predictions.tolist()
                            fin=dict()
                            fin['name']=model_name
                            fin['MAE']=result_set['MAE']
                            fin['MSE']=result_set['MSE']
                            fin['R2 Score']=result_set['R2 Score']
                            fin['RMSE']=result_set['RMSE']
                        except Exception as e:
                            print(e)
                            continue
                    else:
                        try:
                            with self.mlp.run():
                                default_tags=[model_name,"Regression"]
                                tag_list=default_tags+self.tags
                                self.mlp.set_tags(tag_list)
                                model=model
                                model.fit(trainx, trainy)
                                predictions=model.predict(testx)
                                predictions=predictions.tolist()
                            
                                # log performace metrics
                                self.mlp.log_metric("R2 Score", r2_score(testy,predictions))
                                self.mlp.log_metric("MAE", mean_absolute_error(testy,predictions))
                                self.mlp.log_metric("MSE", mean_squared_error(testy,predictions))
                                self.mlp.log_metric("RMSE", mean_squared_error(testy,predictions,squared=False))
                                if self.register==True:
                                    # Save train data and test data
                                    self.mlp.register_artifact("train", trainx)
                                    self.mlp.register_artifact("test", testx,artifact_type="testing")
                                # Save the model
                                if self.register_model==True:
                                        self.mlp.scikit_learn.register_model(model_name, model)
                                result1=self.mlp.get_info()
                            
                                fin=dict()
                                fin['name']=model_name
                                fin['MAE']=result1['metrics']['MAE']
                                fin['MSE']=result1['metrics']['MSE']
                                fin['RMSE']=result1['metrics']['RMSE']
                                fin['R2 Score']=result1['metrics']['R2 Score']
                        except Exception as e:
                            print (e)
                            continue    
                    
                    prediction_set[model_name]=predictions  
                    result=result.append(fin, ignore_index=True)    
               
        if self.task=='classification' or self.metric=='R2 Score': 
            result.sort_values(by=self.metric,ascending=False,inplace=True)
        else:
            result.sort_values(by=self.metric,ascending=True,inplace=True)
            
        if tune_best==False:
            return prediction_set,result
        else:
            result=result.head(1)
            best_model_name=str(result.name.values[0])
            
            prediction_set,result=self.param_tune_model(trainx=trainx,testx=testx,trainy=trainy,testy=testy,model_tune=best_model_name)
            return prediction_set,result

    def param_tune_model(self,model_tune,trainx,testx,trainy,testy):
       
        self.mlp.set_experiment(self.exp_name)    
        best_model_name=model_tune
        
        with self.mlp.run():
            if self.task=="classification":
                default_tags=["Hyper-param-tuning-clf",best_model_name]
                tag_list=default_tags+self.tags
                self.mlp.set_tags(tag_list)
                final_model=self.classification_models[best_model_name].fit(trainx, trainy)
                if self.metric=='accuracy': score= make_scorer(accuracy_score,average='weighted')
                elif self.metric=='recall': score=make_scorer(recall_score,average='weighted')
                elif self.metric=='precision': score=make_scorer(precision_score,average='weighted')
                else: score=make_scorer(f1_score,average='weighted')
                CV_cfl = GridSearchCV(estimator = final_model, param_grid = self.param_grid[best_model_name], scoring= score, cv=3, verbose = 2)
                CV_cfl.fit(trainx, trainy)
                self.mlp.log_params(CV_cfl.best_params_)
                predictions=CV_cfl.best_estimator_.predict(testx)

                result_set={
                    "accuracy": accuracy_score(testy,predictions),
                    "precision": precision_score(testy,predictions,average='macro'),
                    "recall": recall_score(testy,predictions,average='macro'),
                    "f1_score": f1_score(testy,predictions,average='macro')}
                self.mlp.log_metrics(result_set)
            elif self.task=="regression":
                default_tags=["Hyper-param-tuning-reg",best_model_name]
                tag_list=default_tags+self.tags
                self.mlp.set_tags(tag_list)
                final_model=self.regression_models[best_model_name].fit(trainx, trainy)
                if self.metric=='MSE': score= 'neg_mean_squared_error'
                elif self.metric=='MAE': score='neg_mean_absolute_error'
                elif self.metric=='R2 Score': score='r2'
                else: score='neg_root_mean_squared_error'
                CV_cfl = GridSearchCV(estimator = final_model, param_grid = self.param_grid[best_model_name], scoring=score,cv=3,verbose = 2)
                CV_cfl.fit(trainx, trainy)
                self.mlp.log_params(CV_cfl.best_params_)
                predictions=CV_cfl.best_estimator_.predict(testx)

                result_set={
                    "MSE": mean_squared_error(testy,predictions),
                    "MAE": mean_absolute_error(testy,predictions),
                    "R2 Score": r2_score(testy,predictions),
                    "RMSE": mean_squared_error(testy,predictions,squared=False)}
                self.mlp.log_metrics(result_set)
            if self.register_model==True:
                self.mlp.scikit_learn.register_model(best_model_name, CV_cfl.best_estimator_)
            if self.register==True:
                # Save train data and test data
                self.mlp.register_artifact("train", trainx)
                self.mlp.register_artifact("test", testx,artifact_type="testing")

            return predictions,result_set
        
       