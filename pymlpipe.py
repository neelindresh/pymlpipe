import os
from pymlpipe.utils.database import create_folder
import uuid
import yaml
from contextlib import contextmanager
import pandas as pd
import shutil
import pickle
import sklearn
import datetime
    
    
class Context_Manager:
        def __init__(self,name,feature_store,run_id=None):
            super(PyMLPipe)
            if run_id==None:
                self.runid = str(uuid.uuid4())
            else:
                self.runid=run_id
            self.name = name
            self.feature_store=feature_store
            self.exp_path=os.path.join(self.feature_store,self.name,self.runid)
            self.folders={"artifacts":os.path.join(self.exp_path,"artifacts"),
                          "metrics":os.path.join(self.exp_path,"metrics"),
                          "models":os.path.join(self.exp_path,"models"),
                          "params":os.path.join(self.exp_path,"params")}
            self.info_dict=[]
        def get_path(self):
            return self.exp_path
            
        def structure(self):
            self.exp_path=create_folder(self.feature_store,self.name)
            self.exp_path=create_folder(self.exp_path,self.runid)
            self._create_all_folders(self.exp_path)
            return self.exp_path
        
        def _create_all_folders(self,exp_path):
            for i in self.folders:
                create_folder(exp_path,i)
        def write_to_yaml(self,info):
            with open(os.path.join(self.exp_path,"info.yaml"), 'w') as file:
                documents = yaml.dump(info, file)
            
        
        
class PyMLPipe:
    def __init__(self):
        
        self.feature_store=create_folder(os.getcwd())
        self.experiment_name='0'
        self.folders=None
        self.experiment_path=None
        self.info={}
        
        
    @contextmanager
    def run(self,experiment_name=None,runid=None):
        """_summary_: start a context manager for with statement
        1. When run is started it will create
            a. RUN ID
            b. EXPERIMENT ID
            c. FOLDERS for storing the details

        Args:
            experiment_name (str, optional): gives a experiment name. Defaults to None.
            runid (str, optional): gives a runid. Defaults to None.

        Returns:
            class context_run(object): object for the context manager
        """
        if experiment_name!=None:
            self.experiment_name=experiment_name
        r=Context_Manager(self.experiment_name,
                           self.feature_store,runid)
        
        self._write_info_run(self.experiment_name, r.runid)
        r.structure()
        self.context_manager=r
        #initialize models 
        self.scikit_learn=ScikitLearn(self.context_manager.folders)
        #self.pytorch=Pytorch(self.context_manager.folders)
        yield r
        self.info["execution_time"]=str(datetime.datetime.now()).split(".")[0]
        
        self.info["model"]={"model_name":self.scikit_learn.model_name,"model_path":self.scikit_learn.model_path}
        self.context_manager.write_to_yaml(self.info)
        
    
    def set_experiment(self,name):
        self.experiment_name=name
        exp_path=create_folder(self.feature_store,self.experiment_name)
        self._write_info_experiment(name,exp_path)
    
    
    def set_tag(self):
        if "tags" not in self.info:
            self.info["tags"]=[]
        if isinstance(tag_dtag_valueict,dict) or isinstance(tag_dtag_valueict,list) or isinstance(tag_dtag_valueict,set): 
           raise TypeError("unsupported type, Expected 'str','int','float' got "+str(type(tag_dict)))
        self.info["tags"].append(tag_value)
        
    
    
    def set_tags(self,tag_dict:list):
        if "tags" not in self.info:
            self.info["tags"]=[]
        if isinstance(tag_dict,list): 
            self.info["tags"].extend(tag_dict)
        else:
            raise TypeError("unsupported type, Expected 'list' got "+str(type(tag_dict)))
        
    def get_tags(self):
        return self.info["tags"]
        
    def set_version(self,version):
        if isinstance(version,dict) or isinstance(version,list) or isinstance(version,set): 
           raise TypeError("unsupported type, Expected 'str','int','float' got "+str(type(tag_dict)))
        self.info["version"]=version
        
        
    def get_version(self):
        return self.info["version"]
    
    def log_metrics(self,metric_dict:dict):
        if "metrics" not in self.info:
            self.info["metrics"]={}
        if isinstance(metric_dict,dict): 
            self.info["metrics"].update(metric_dict)
        else:
            raise TypeError("unsupported type, Expected 'dict' got "+str(type(metric_dict)))
        
        
    def log_matric(self,metric_name,metric_value):
        if "metrics" not in self.info:
            self.info["metrics"]={}
        mv=None
        if not isinstance(metric_value,int) and not isinstance(metric_value,float): 
            raise TypeError("unsupported type, 'metric_value' Expected 'int','float' got "+str(type(metric_value)))
        if not isinstance(metric_name,str): 
            raise TypeError("unsupported type, 'metric_value' Expected 'str' got "+str(type(metric_name)))
        self.info["metrics"][metric_name]=metric_value
       
        
    def log_params(self,param_dict:dict):
        if "params" not in self.info:
            self.info["params"]={}
        if isinstance(param_dict,dict): 
            self.info["params"].update(param_dict)
        else:
            raise TypeError("unsupported type, Expected 'dict' got "+str(type(metric_dict)))
        
        
    def log_param(self,param_name,param_value):
        if "params" not in self.info:
            self.info["params"]={}
        mv=None
        if not isinstance(param_value,int) and not isinstance(param_value,float) and not isinstance(param_value,str): 
            raise TypeError("unsupported type, 'param_value' Expected 'int','float','str) got "+str(type(metric_value)))
        if not isinstance(param_name,str): 
            raise TypeError("unsupported type, 'param_name' Expected 'str' got "+str(type(metric_name)))
        self.info["params"][param_name]=param_value
    
    def register_artifact(self,artifact_name,artifact):
        if not isinstance(artifact, pd.DataFrame):
            raise TypeError("Please provide DataFrame in 'artifact'")
        if artifact_name=="" or artifact_name==None:
            raise ValueError("Please provide a name in 'artifact_name' which is not '' or None")
        path=os.path.join(self.context_manager.folders["artifacts"],artifact_name)
                          
        artifact.to_csv(path,index=False)
        if "artifact" not in self.info:
            self.info["artifact"]=[]
        self.info["artifact"].append(path)
        
        
    def register_artifact_with_path(self,artifact):
        if not isinstance(artifact, str):
            raise TypeError("Please provide full path of artifact")
        if not os.path.exists(artifact):
            raise ValueError("Please provide correct path of artifact")
        
        shutil.copy(artifact, self.context_manager.folders["artifacts"])
        if "artifact" not in self.info:
            self.info["artifact"]=[]
        self.info["artifact"].append(os.path.join(self.context_manager.folders["artifacts"],os.path.basename(artifact)))
        
        
    def get_info(self):
        return self.info 
    
    
    def get_artifact(self):
        return self.info["artifact"]
    
    def _write_info_experiment(self,experiment_name,path):
        fulllist={}
        if os.path.exists(os.path.join(self.feature_store,"experiment.yaml")):
            with open(os.path.join(self.feature_store,"experiment.yaml")) as file:
                fulllist = yaml.load(file, Loader=yaml.FullLoader)
            if experiment_name not in fulllist:
                fulllist[experiment_name]={"experiment_path":path,
                                           "runs":[],
                                           "execution_time":str(datetime.datetime.now()).split(".")[0]
                                           }
            else:
                fulllist[experiment_name]["execution_time"]=str(datetime.datetime.now()).split(".")[0]
        else:
            fulllist[experiment_name]={"experiment_path":path,
                                       "runs":[],
                                       "execution_time":str(datetime.datetime.now()).split(".")[0]
                                       }
        
            
        with open(os.path.join(self.feature_store,"experiment.yaml"), 'w') as file:
                documents = yaml.dump(fulllist, file)
                
    
    def _write_info_run(self,experiment_name,run_id):
        fulllist={}
        
        with open(os.path.join(self.feature_store,"experiment.yaml")) as file:
            fulllist = yaml.load(file, Loader=yaml.FullLoader)
        fulllist[experiment_name]["runs"].append(run_id)
            
        
            
        with open(os.path.join(self.feature_store,"experiment.yaml"), 'w') as file:
                documents = yaml.dump(fulllist, file)
        
    
    def set_uri(self):
        pass
    

                
#explainer*
#https://github.com/SauceCat/PDPbox             
 #https://github.com/AustinRochford/PyCEbox               


class ScikitLearn:
    def __init__(self,folders):
        self.folders=folders
        self.model_name=""
        self.model_path=""
        
        
    def register_model(self,model_name,model):
        if "sklearn" in str(type(model)):
            
            pickle.dump(model, open(os.path.join(self.folders["models"],model_name+'.pkl'), 'wb'))
        else:
            raise TypeError("Error:Expected ScikitLearn Module!!!!")
        self.model_name=model_name
        self.model_path=os.path.join(self.folders["models"],model_name+'.pkl')
    

class Pytorch:
    def __init__(self):
        pass
    def register_model(self):
        pass