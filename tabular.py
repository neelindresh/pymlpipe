import os
from pymlpipe.utils.database import create_folder
from pymlpipe.utils.getschema import schema_

import uuid
import yaml
from contextlib import contextmanager
import pandas as pd
import shutil
import pickle
import sklearn
import datetime
    
    
class Context_Manager:
        """_summary_: Context Manager for with statement
        1. creates folders and subfolders
        2. creates runid for a run instance
        """
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
            """_summary_

            Returns:
                _type_: _description_
            """
            return self.exp_path
            
        def structure(self):
            """_summary_

            Returns:
                _type_: _description_
            """
            self.exp_path=create_folder(self.feature_store,self.name)
            self.exp_path=create_folder(self.exp_path,self.runid)
            self._create_all_folders(self.exp_path)
            return self.exp_path
        
        def _create_all_folders(self,exp_path):
            """_summary_

            Args:
                exp_path (_type_): _description_
            """
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
        self.info["tags"]=[]
        self.info["metrics"]={}
        self.info["params"]={}
        self.info["artifact"]=[]
        self.info["model"]={}
        self.info["artifact_schema"]=[]
        
        
    def __reset__(self):
        self.feature_store=create_folder(os.getcwd())
        self.folders=None
        self.experiment_path=None
        
        self.info["tags"]=[]
        self.info["metrics"]={}
        self.info["params"]={}
        self.info["artifact"]=[]
        self.info["model"]={}
        self.info["artifact_schema"]=[]
        
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
        if self.scikit_learn.registered:
            self.info["model"]={"model_name":self.scikit_learn.model_name,
                                "model_path":self.scikit_learn.model_path,
                                "model_params": self.scikit_learn.model_params,
                                "model_class":self.scikit_learn.model_class,
                                "model_type":self.scikit_learn.model_type,
                                "model_tags":self.scikit_learn.model_tags,
                                "registered":self.scikit_learn.registered
                                }
        #print(self.info)
        self.context_manager.write_to_yaml(self.info)
        self.__reset__()
        
    
    def set_experiment(self,name):
        """_summary_: sets the experiment name

        Args:
            name (str): name of the experiment
        """
        self.experiment_name=name
        exp_path=create_folder(self.feature_store,self.experiment_name)
        self._write_info_experiment(name,exp_path)
    
    
    def set_tag(self,tag_value):
        """_summary_: sets a tag for a perticular run
        Args:
            name (str or int or float): tag name 
        Raises:
            TypeError: Supported type 'str','int','float'
        """
        
            
        if isinstance(tag_value,dict) or isinstance(tag_value,list) or isinstance(tag_value,set): 
           raise TypeError("unsupported type, Expected 'str','int','float' got "+str(type(tag_value)))
        self.info["tags"].append(tag_value)
        
    
    
    def set_tags(self,tag_dict:list):
        """_summary_:sets N no of tags for a perticular run

        Args:
            tag_dict (list): tag names in list format

        Raises:
            TypeError: Expected 'list'
        """
       
        if isinstance(tag_dict,list): 
            self.info["tags"].extend(tag_dict)
        else:
            raise TypeError("unsupported type, Expected 'list' got "+str(type(tag_dict)))
        
    def get_tags(self):
        """_summary_: get all the tags that are associated with the run

        Returns:
            list: tags that are associated with the run
        """
        return self.info["tags"]
        
    def set_version(self,version):
        """_summary_:sets version number for the perticular run

        Args:
            version (str or int or float): version number

        Raises:
            TypeError: Expected 'str','int','float'
        """
        if isinstance(version,dict) or isinstance(version,list) or isinstance(version,set): 
           raise TypeError("unsupported type, Expected 'str','int','float' got "+str(type(tag_dict)))
        self.info["version"]=version
        
        
    def get_version(self):
        """_summary_:get the version number associated with the run


        Returns:
            _type_: version number
        """
        return self.info["version"]
    
    def log_metrics(self,metric_dict:dict):
        """_summary_: log metrics for the model run

        Args:
            metric_dict (dict): key value pair with metric name and metric value

        Raises:
            TypeError: Expected 'dict'
        """
           
        if isinstance(metric_dict,dict): 
            self.info["metrics"].update({i:float("{0:.2f}".format(j)) for i,j in metric_dict.items()})
        else:
            raise TypeError("unsupported type, Expected 'dict' got "+str(type(metric_dict)))
        
        
    def log_metric(self,metric_name,metric_value):
        """_summary_: log single metric for the model run

        Args:
            metric_name (str): name of the metric
            metric_value (int or float): value of the metric

        Raises:
            TypeError: metric_name expected to be str
            TypeError: metric_value expected to be int or float
        """
        
        mv=None
        if not isinstance(metric_value,int) and not isinstance(metric_value,float): 
            raise TypeError("unsupported type, 'metric_value' Expected 'int','float' got "+str(type(metric_value)))
        if not isinstance(metric_name,str): 
            raise TypeError("unsupported type, 'metric_value' Expected 'str' got "+str(type(metric_name)))
        
        
        self.info["metrics"][metric_name]=float("{0:.2f}".format(metric_value))
       
        
    def log_params(self,param_dict:dict):
        """_summary_: log parameters for the model run

        Args:
            param_dict (dict): key value pair with parameter name and parameter value

        Raises:
            TypeError: Expected 'dict'
        """
        
            
        if isinstance(param_dict,dict): 
            self.info["params"].update(param_dict)
        else:
            raise TypeError("unsupported type, Expected 'dict' got "+str(type(metric_dict)))
        
        
    def log_param(self,param_name,param_value):
        """_summary_:log single parameter for the model run

        Args:
            param_name (str): _description_
            param_value (int or float or str): _description_

        Raises:
            TypeError: param_name Expected 'str' 
            TypeError: param_value Expected 'int','float','str' 
        """
       
        mv=None
        if not isinstance(param_value,int) and not isinstance(param_value,float) and not isinstance(param_value,str): 
            raise TypeError("unsupported type, 'param_value' Expected 'int','float','str' got "+str(type(metric_value)))
        if not isinstance(param_name,str): 
            raise TypeError("unsupported type, 'param_name' Expected 'str' got "+str(type(metric_name)))
        self.info["params"][param_name]=param_value
    
    def register_artifact(self,artifact_name,artifact,artifact_type="training"):
        """_summary_: Save Artifact as part of data verion control

        Args:
            artifact_name (str): name of the artifact
            artifact (pandas DataFrame): pandas DataFrame object with the data
            artifact_type (str, optional): Defaults to "training". artifact_type can be [training,testing,validation,dev,prod]

        Raises:
            TypeError: Expected DataFrame object
            ValueError: artifact_name should have a string value
        """
        if not isinstance(artifact, pd.DataFrame):
            raise TypeError("Please provide DataFrame in 'artifact'")
        if artifact_name=="" or artifact_name==None:
            raise ValueError("Please provide a name in 'artifact_name' which is not '' or None")
        path=os.path.join(self.context_manager.folders["artifacts"],artifact_name)
        dataschema=artifact.describe(include='all')
        
        artifact.to_csv(path,index=False)
        
        
            
        
        self.info["artifact"].append({
            "name":artifact_name,
            "path":path,
            "tag":artifact_type
        })
        schema_data,schema_details=schema_(artifact)
        self.info["artifact_schema"].append({
                "name":artifact_name,
                "schema":schema_data,
                "details":schema_details
            }
        )
        
        
    def register_artifact_with_path(self,artifact,artifact_type="training"):
        """_summary_

        Args:
            artifact (str): path of the artifact
            artifact_type (str, optional): _description_. Defaults to "training".artifact_type can be [training,testing,validation,dev,prod]

        Raises:
            TypeError: artifact path should be str
            ValueError: artifact path should be correct
        """
        if not isinstance(artifact, str):
            raise TypeError("Please provide full path of artifact")
        if not os.path.exists(artifact):
            raise ValueError("Please provide correct path of artifact")
        
        shutil.copy(artifact, self.context_manager.folders["artifacts"])
        
        path=os.path.join(self.context_manager.folders["artifacts"],os.path.basename(artifact))
        self.info["artifact"].append({
            "name":os.path.basename(path),
            "path":path,
            "tag":artifact_type
        })
        filename=os.path.basename(artifact)
        if filename.endswith('.csv'):
            artifact=pd.read_csv(path)
        elif filename.endswith('.xlxs'):
            artifact=pd.read_excel(path)
        elif filename.endswith('.parquet'):
            artifact=pd.read_parquet(path)
        else:
            print("Error: Unknown file type cannot generate Schema!!!!")
            return
        
        schema_data,schema_details=schema_(artifact)
        self.info["artifact_schema"].append({
                "name":filename,
                "schema":schema_data,
                "details":schema_details
            }
        )
        
    def get_info(self):
        """_summary_: get the whole run details

        Returns:
            dict: information about the whole run
        """
        return self.info 
    
    
    def get_artifact(self):
        """_summary_: get the artifact details

        Returns:
            dict: returns the artifact detail
        """
        return self.info["artifact"]
    
    def _write_info_experiment(self,experiment_name,path):
        """_summary_: writes to the experiment schema

        Args:
            experiment_name (str): name of the experiment
            path (str): path to save the run details
        """
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
        """_summary_:writes to the run schema

        Args:
            experiment_name (str): name of the experiment
            run_id (str): ID for the running instance
        """
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
        self.model_class=""
        self.model_type=""
        self.model_params={}
        self.model_tags={}
        self.registered=False
        
        
    def register_model(self,model_name,model):
        if "sklearn" in str(type(model)):
            
            pickle.dump(model, open(os.path.join(self.folders["models"],model_name+'.pkl'), 'wb'))
        else:
            raise TypeError("Error:Expected ScikitLearn Module!!!!")
        self.model_name=model_name
        self.model_path=os.path.join(self.folders["models"],model_name+'.pkl')
        self.model_class=type(model).__name__
        self.model_params=model.get_params()
        self.model_tags={tag:str(value) for tag,value in model._get_tags().items()}
        self.model_type="scikit-learn"
        self.registered=True
    