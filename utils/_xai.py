import matplotlib.pyplot as pl
from sklearn import inspection
import shap
import pandas as pd
import os
import numpy as np



XAI_MAP={
    "TreeBasedModels": ['BaseDecisionTree',
 'DecisionTreeClassifier',
 'DecisionTreeRegressor',
 'ExtraTreeClassifier',
 'ExtraTreeRegressor',
 'BaseEnsemble',
 'RandomForestClassifier',
 'RandomForestRegressor',
 'RandomTreesEmbedding',
 'ExtraTreesClassifier',
 'ExtraTreesRegressor',
 'BaggingClassifier',
 'BaggingRegressor',
 'IsolationForest',
 'GradientBoostingClassifier',
 'GradientBoostingRegressor',
 'AdaBoostClassifier',
 'AdaBoostRegressor',
 'VotingClassifier',
 'VotingRegressor',
 'StackingClassifier',
 'StackingRegressor',
 "XGBClassifier",
 "XGBRegressor",
 "CatBoostClassifier",
 "CatBoostRegressor",
 "LGBMClassifier",
 "LGBMRegressor"
 ],
"LinearModels": ['ARDRegression',
 'BayesianRidge',
 'ElasticNet',
 'ElasticNetCV',
 'Hinge',
 'Huber',
 'HuberRegressor',
 'Lars',
 'LarsCV',
 'Lasso',
 'LassoCV',
 'LassoLars',
 'LassoLarsCV',
 'LassoLarsIC',
 'LinearRegression',
 'LogisticRegression',
 'LogisticRegressionCV',
 'ModifiedHuber',
 'MultiTaskElasticNet',
 'MultiTaskElasticNetCV',
 'MultiTaskLasso',
 'MultiTaskLassoCV',
 'OrthogonalMatchingPursuit',
 'OrthogonalMatchingPursuitCV',
 'PassiveAggressiveClassifier',
 'PassiveAggressiveRegressor',
 'Perceptron',
 'Ridge',
 'RidgeCV',
 'RidgeClassifier',
 'RidgeClassifierCV',
 'SGDClassifier',
 'SGDRegressor',
 'SquaredLoss',
 'TheilSenRegressor',
 'RANSACRegressor',
 'PoissonRegressor',
 'GammaRegressor',
 'TweedieRegressor'],
}

class Explainer():
    def __init__(self,model,data,artifact_path):
        self.model=model
        self.data=data
        self.artifact_path=artifact_path
        self.feature_map=self.data.columns
    def explain(self):
        model_class=type(self.model)
        model_name=type(self.model).__name__
        flag=False
        if model_name in XAI_MAP["LinearModels"]:
            self.coef_based_feature_importance(self.model,np.std(self.data,0),self.feature_map,os.path.join(self.artifact_path,"explainer"))
            try:
                self.tree_linear_summary_plot(self.model,self.data,self.feature_map,os.path.join(self.artifact_path,"explainer"))
            except Exception as e:
                flag=True
                print("Warning:Instance of model {model} not supported".format(model=model_name))
            
        elif model_name in XAI_MAP["TreeBasedModels"]:
            self.tree_based_feature_importance(self.model,self.feature_map,os.path.join(self.artifact_path,"explainer"))
            try:
                self.tree_expainer_summary_plot(self.model,self.data,self.feature_map,os.path.join(self.artifact_path,"explainer"))
            except Exception as e:
                flag=True
                print("Warning: Instance of model {model} not supported".format(model=model_name))
            
        else:
            #implement XAI for NeuralNetworks
            pass
        if not flag:
            return {
                "feature_explainer":os.path.join(self.artifact_path,"explainer.csv"),
                "shap":os.path.join(self.artifact_path,"explainer.svg")
                }
        else:
            return {
                "feature_explainer":os.path.join(self.artifact_path,"explainer.csv"),
                "shap":""
            }

    def tree_expainer_summary_plot(self,model,xtrain,feature_map,fig_name):
        shap_xgb_explainer = shap.TreeExplainer(model)
        shap_xgb_values_train = shap_xgb_explainer.shap_values(xtrain)
        shap.summary_plot(shap_xgb_values_train, xtrain,feature_names=feature_map,show=False)
        pl.savefig("{fig_name}.svg".format(fig_name=fig_name),dpi=700,bbox_inches='tight')
        pl.close('all')
        
        
    def tree_linear_summary_plot(self,model,xtrain,feature_map,fig_name):
        shap_xgb_explainer = shap.LinearExplainer(model,xtrain)
        shap_xgb_values_train = shap_xgb_explainer.shap_values(xtrain)
        shap.summary_plot(shap_xgb_values_train, xtrain,feature_names=feature_map,show=False)
        pl.savefig("{fig_name}.svg".format(fig_name=fig_name),dpi=700,bbox_inches='tight')
        pl.close('all')

    def permutation_feature_importance(self,model,trainx,trainy):
        permutation_imp=inspection.permutation_importance(model, trainx, trainy, n_jobs=-1,scoring='accuracy', n_repeats=8,)
        return permutation_imp.importances_mean


    def tree_based_feature_importance(self,model,feature_map,path):
        model_ranks=pd.DataFrame([{"feature":f,"importance":fi} for f,fi in zip(feature_map,model.feature_importances_)])
        dt_rank_df = pd.DataFrame({"feature":model_ranks["feature"],"importance":model_ranks["importance"],'rank': model_ranks["importance"].rank(method='first', ascending=False).astype(int)})
        dt_rank_df.to_csv('{path}.csv'.format(path=path),index=False)

    def coef_based_feature_importance(self,model,std,feature_map,path):
        maps={"feature":feature_map,}
        n_coff=0
        for idx,i in enumerate(model.coef_):
            maps["coef_norm_"+str(idx)]=model.coef_[idx] *std
            n_coff+=1
        df=pd.DataFrame(maps)
        df=df.round(3)
        df["avg_coef_norm"]=df.sum(axis=1)/n_coff
        
        ndf=df.sort_values(by="avg_coef_norm",ascending=False)
        ndf.to_csv('{path}.csv'.format(path=path),index=False)

    def permuatation_feature_importance(self,model,test_x,test_y,feature_map):
        importancef=inspection.permutation_importance(model,test_x,test_y,n_jobs=-1, n_repeats=8)
        return pd.DataFrame([{"feature":f,"importance":fi} for f,fi in zip(feature_map,importancef.importances_mean)])




