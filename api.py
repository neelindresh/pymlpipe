import os
from pymlpipe.utils import yamlio
from pymlpipe.utils import factory
import pandas as pd
class Client:
    def __init__(self,path:str=None):
        if path:
            self.path=path
        elif "modelrun" in os.listdir():
            self.path=os.path.join(os.getcwd(),"modelrun")
            
            print(f"No Path specified, defaulting to current path {self.path}")
            
    def get_all_experiments(self):
        all_experiments=yamlio.read_yaml(
            os.path.join(self.path, factory.DEFAULT["ModelRunInfo"])
        )
        return list(all_experiments.keys())
    
    def get_all_run_ids(self,experiment_name):
        all_tunids=yamlio.read_yaml(
            os.path.join(self.path, factory.DEFAULT["ModelRunInfo"])
        )
        return all_tunids[experiment_name]["runs"]
    
    def get_run_details(self,experiment_name,runid):
        return yamlio.read_yaml(
            os.path.join(
                self.path, experiment_name, runid, factory.DEFAULT["RunInfo"]
            )
        )
    def get_all_run_details(self,experiment_name):
        all_runids=yamlio.read_yaml(
            os.path.join(self.path, factory.DEFAULT["ModelRunInfo"])
        )
        all_paths={id : os.path.join(
                self.path, experiment_name, id, factory.DEFAULT["RunInfo"]
            ) for id in all_runids[experiment_name]["runs"]}
        return {
            
            id: yamlio.read_yaml(path) for id,path in all_paths.items()
        }
    def get_metrics_comparison(self,experiment_name:str,format:str=None,sort_by:str=None,with_version=False):
        all_runids=yamlio.read_yaml(
            os.path.join(self.path, factory.DEFAULT["ModelRunInfo"])
        )
        all_paths={id : os.path.join(
                self.path, experiment_name, id, factory.DEFAULT["RunInfo"]
            ) for id in all_runids[experiment_name]["runs"]}
        data={

            id: yamlio.read_yaml(path) for id,path in all_paths.items()
        }
        comparison={}
        for id,d in data.items():
            comparison[id]={
                "model": d["model"]["model_class"],
                
                }
            if with_version: comparison[id]["version"]=d["version"] 
            comparison[id].update(d["metrics"])
        if format:
            if sort_by:
                return pd.DataFrame(comparison).T.sort_values(by=sort_by,ascending=False)
            return pd.DataFrame(comparison).T
        else:
            comparison
            
    def get_model_details(self,experiment_name,runid,format:str=None):
        data=yamlio.read_yaml(
            os.path.join(
                self.path, experiment_name, runid, factory.DEFAULT["RunInfo"]
            )
        )
        model_details=data["model"]
        _exceptions_=["model_params","model_tags"]
        print(model_details.keys())
        model_info=[]
        model_info.extend(
            {"name": model_detail, "value": model_details[model_detail]}
            for model_detail in model_details
            if model_detail not in _exceptions_
        )
        model_params = [
            {"name": params, "value": model_details["model_params"][params]}
            for params in model_details["model_params"]
        ]
        model_tags = [
            {"name": params, "value": model_details["model_tags"][params]}
            for params in model_details["model_tags"]
        ]
        if format:
            return pd.DataFrame(model_info),pd.DataFrame(model_params),pd.DataFrame(model_tags)
        else:
            return model_info,model_params,model_tags
            
        