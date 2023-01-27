from sklearn.datasets import  load_iris,load_diabetes
import pandas as pd
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
#import PyMLPipe from tabular 
from pymlpipe.tabular import PyMLPipe
from sklearn.model_selection import GridSearchCV


class AutoMLPipe():
    def __init__(self,exp_name,task,metric,data,label,tags=[],tune=True,tune_best=False,test_size=0.20,version=1.0,register_model=False,register_artifacts=False,exclude=[]):
        '''
        exp_name: name of experiment
        task: regression/classification
        metric: for classification -> accuracy,recall,precision, f1/ for regression -> MAE,MSE,RMSE,R2 Score
        data: data on which the model to be fit
        label: target variable
        tags: list of custom-tags for the run
        tune: param tune all the models
        tune_best: param tune the best model
        test_size: size of test dataset
        version: experiment version
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
        self.mlp=PyMLPipe()
        self.register_model=register_model
        self.register=register_artifacts
        self.tags=tags
        self.tune=tune
        self.tune_best=tune_best
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
            'max_depth': [None,2,3],
           'learning_rate': [0.01,0.1]
        }   

    def run_automl(self):

        # Set Experiment name
        self.mlp.set_experiment(self.exp_name)
        # Set Version name
        self.mlp.set_version(self.version)
        trainx,testx,trainy,testy=train_test_split(self.data,self.label,test_size=self.test_size)
        
        
        result=pd.DataFrame()
        if self.task=='classification':
            for model_name,model in self.classification_models.items():
                if model_name not in self.exclude:
                        with self.mlp.run():
                            # set tags
                            default_tags=[model_name,"Classification"]
                            tag_list=default_tags+self.tags
                            self.mlp.set_tags(tag_list)
                            if self.tune==True:
                                
                                    final_model=model.fit(trainx, trainy)
                                    if self.metric=='accuracy': score= make_scorer(accuracy_score,average='weighted')
                                    elif self.metric=='recall': score=make_scorer(recall_score,average='weighted')
                                    elif self.metric=='precision': score=make_scorer(precision_score,average='weighted')
                                    else: score=make_scorer(f1_score,average='weighted')
                                    model = GridSearchCV(estimator = final_model, param_grid = self.param_grid[model_name], scoring=score,verbose = 2)
                                    model.fit(trainx, trainy)
                                    predictions=model.best_estimator_.predict(testx)
                            
                            else:
                           
                                model=model
                                model.fit(trainx, trainy)
                                predictions=model.predict(testx)

                            self.mlp.log_metric("Accuracy", accuracy_score(testy,predictions))
                            self.mlp.log_metric("Precision", precision_score(testy,predictions,average='macro'))
                            self.mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
                            self.mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
                            if self.register==True:
                                # Save train data and test data
                                self.mlp.register_artifact("train", trainx)
                                self.mlp.register_artifact("test", testx,artifact_type="testing")
                            if self.register_model==True:
                                 self.mlp.scikit_learn.register_model(model_name, model)
                            
                            result1=self.mlp.get_info()
                            fin=dict()
                            fin['name']=model_name
                            fin['accuracy']=result1['metrics']['Accuracy']
                            fin['precision']=result1['metrics']['Precision']
                            fin['recall']=result1['metrics']['Recall']
                            fin['f1']=result1['metrics']['F1']
                            result=result.append(fin,ignore_index=True)
        elif self.task=='regression':
            for model_name,model in self.regression_models.items():
                if model_name not in self.exclude:
                        with self.mlp.run():
                            # set tags
                            default_tags=[model_name,"Regression"]
                            tag_list=default_tags+self.tags
                            self.mlp.set_tags(tag_list)
                            if self.tune==True:
                                final_model=model.fit(trainx, trainy)
                                if self.metric=='MSE': score= 'neg_mean_squared_error'
                                elif self.metric=='MAE': score='neg_mean_absolute_error'
                                elif self.metric=='R2 Score': score='r2'
                                else: score='neg_root_mean_squared_error'
                                model = GridSearchCV(estimator = final_model, param_grid = self.param_grid[model_name], scoring=score,verbose = 2)
                                model.fit(trainx, trainy)
                                predictions=model.best_estimator_.predict(testx)
                            else:
                                model=model
                                model.fit(trainx, trainy)
                                predictions=model.predict(testx)
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
                            #fin['runid']=self.mlp.runid()
                            #result=result.append(result1,ignore_index=True)
                            result=result.append(fin, ignore_index=True)    
               
        if self.task=='classification' or self.metric=='R2 Score': 
            result.sort_values(by=self.metric,ascending=False,inplace=True)
        else:
            result.sort_values(by=self.metric,ascending=True,inplace=True)
            
        if self.tune_best==False:
            return result
        else:
            result=result.head(1)
            best_model_name=str(result.name.values[0])
            
            self.param_tune_model(model_tune=best_model_name)
       

    def param_tune_model(self,model_tune):
       
        self.mlp.set_experiment(self.exp_name)    
        best_model_name=model_tune
        
        with self.mlp.run():
            trainx,testx,trainy,testy=trainx,testx,trainy,testy=train_test_split(self.data,self.label,test_size=self.test_size)
            print(best_model_name)
            if self.task=="classification":
                default_tags=["Hyper-param-tuning-clf",best_model_name]
                tag_list=default_tags+self.tags
                self.mlp.set_tags(tag_list)
                final_model=self.classification_models[best_model_name].fit(trainx, trainy)
                if self.metric=='accuracy': score= make_scorer(accuracy_score,average='weighted')
                elif self.metric=='recall': score=make_scorer(recall_score,average='weighted')
                elif self.metric=='precision': score=make_scorer(precision_score,average='weighted')
                else: score=make_scorer(f1_score,average='weighted')
                CV_cfl = GridSearchCV(estimator = final_model, param_grid = self.param_grid[best_model_name], scoring= score, verbose = 2)
                CV_cfl.fit(trainx, trainy)
                self.mlp.scikit_learn.register_model(best_model_name, CV_cfl)
            elif self.task=="regression":
                default_tags=["Hyper-param-tuning-reg",best_model_name]
                tag_list=default_tags+self.tags
                self.mlp.set_tags(tag_list)
                final_model=self.regression_models[best_model_name].fit(trainx, trainy)
                if self.metric=='MSE': score= 'neg_mean_squared_error'
                elif self.metric=='MAE': score='neg_mean_absolute_error'
                elif self.metric=='R2 Score': score='r2'
                else: score='neg_root_mean_squared_error'
                CV_cfl = GridSearchCV(estimator = final_model, param_grid = self.param_grid[best_model_name], scoring=score,verbose = 2)
                CV_cfl.fit(trainx, trainy)
                self.mlp.scikit_learn.register_model(best_model_name, CV_cfl)

            best_parameters = CV_cfl.best_params_
            print(CV_cfl.best_score_)
            #cfl=
            print("The best parameters for using this model is", best_parameters)
            predictions=CV_cfl.best_estimator_.predict(testx)
            print(predictions)
            self.mlp.log_params(CV_cfl.best_params_)
            # log performace metrics
            if self.task=="classification":
                self.mlp.log_metric("Accuracy", accuracy_score(testy,predictions))
                self.mlp.log_metric("Precision", precision_score(testy,predictions,average='macro'))
                self.mlp.log_metric("Recall", recall_score(testy,predictions,average='macro'))
                self.mlp.log_metric("F1", f1_score(testy,predictions,average='macro'))
            elif self.task=="regression":
                self.mlp.log_metric("MAE", mean_absolute_error(testy,predictions))
                self.mlp.log_metric("MSE", mean_squared_error(testy,predictions))
                self.mlp.log_metric("RMSE", mean_squared_error(testy,predictions,squared=False))
                self.mlp.log_metric("R2 Score", r2_score(testy,predictions))

            return predictions
