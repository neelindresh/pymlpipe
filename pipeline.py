
import cloudpickle
from pymlpipe.utils import database,yamlio
import os
import datetime
import traceback,sys
import inspect
##---To Be moved into separate file--##

__FOLDER__="ML_pipelines"


class Node:
    def __init__(self,name, func,path):
        self.name=name
        self.func=func
        self.path=path
        self.save(name, func)

    def save(self,name: str,func) -> None:
        """_summary_: Saves a python Function into Mould file

        Args:
            name (str): Name of the function
            func (Object): Actual python function
        """

        self.name_of_file = f"{self.name}.mld"
        self.filename=os.path.join(self.path,self.name_of_file)
        cloudpickle.dump(func, open(self.filename, "wb"))

class PipeLine:
    def __init__(self,pipeline_name,pipeline_path=None):
        self.path=os.getcwd()
        self.pipeline_name=pipeline_name
        self.pipeline_path = __FOLDER__ if pipeline_path is None else pipeline_path
        database.create_folder(self.path)
        self.base_path=database.create_folder(self.path,self.pipeline_path)
        self.path_pipe=database.create_folder(self.base_path,self.pipeline_name)

        self.dag={"nodes":{},"graph":{},"args_map":{},"node_details":{}}
        self.status_code={0:"Started",1:"Completed",2:"Queued",3:"Failed"}
        self._args_tag="args@"

    def add_node(self,node_name:str,function,input_nodes:list =None,args:dict=None,entry_node:bool=False) -> None:
        """_summary_

        Args:
            node_name (str): Name of the node
            function (_type_): Python function you want to execute
            input_nodes (list, optional): List of nodes that are connected to this node. The connected nodes should return a value which will act as an input to the node . Defaults to None.
            entry_node (bool, optional): boolean flag indicating if this is the starting node(first node). Defaults to False.
            args (list, optional): Run time arguments . Defaults to None.

        Raises:
            ValueError: _description_
            TypeError: _description_
            TypeError: _description_
            TypeError: _description_
        """
        '''Exception Start'''
        if entry_node and "root" in self.dag["graph"]:
            raise ValueError("Error!!! entry_node is already set. Two nodes cannot be Entry Node in DAG.")
        if input_nodes != None and not isinstance(input_nodes,list):
            raise TypeError(
                f"Error!!! 'input_node' expected to be type:list got {type(input_nodes)}"
            )
        if not isinstance(entry_node,bool):
            raise TypeError(
                f"Error!!! 'entry_node' expected to be type:bool got {type(input_nodes)}"
            )
        if not isinstance(node_name,str):
            raise TypeError(
                f"Error!!! 'node_name' expected to be type:str got {type(input_nodes)}"
            )

        '''Exception End'''


        node=Node(node_name,function,self.path_pipe)
        self.dag["nodes"][node_name]={
            'filename':node.name_of_file,
            'root_path':self.pipeline_path,
            'sub_path':self.pipeline_name,
            "edge_nodes":input_nodes,
            "args":args
        }

        _arg_names=inspect.getfullargspec(function).args
        ## Mapping args to the input node
        if args !=None:
            for arg_name in _arg_names:
                if arg_name in args:
                    input_nodes.append(f"{self._args_tag}{arg_name}")
        _mapper = dict(zip(input_nodes,_arg_names)) if len(_arg_names)!=0 else {}
        #print("_mapper:",_mapper)
        self.dag["args_map"][node_name]=_mapper
        if entry_node:
            self.dag["graph"]["root"]=[node_name]
        else:
            for ipnode in input_nodes:
                if ipnode.startswith(self._args_tag):
                    continue
                if ipnode in self.dag["graph"]:
                    self.dag["graph"][ipnode].append(node_name)
                else:
                    self.dag["graph"][ipnode]=[node_name]
        if node_name not in self.dag["graph"]:
            self.dag["graph"][node_name]=[]
        self.dag["node_details"][node_name]={"status":self.status_code[2],"start_time":"-","end_time":"-","log":""}


    def register_dag(self):
        """_summary_: Registers the pipeline as an Dag Object
        """
        path_to_yaml=self.path_pipe
        file_name = f"{self.pipeline_name}.yaml"
        
        data=yamlio.read_yaml(os.path.join(self.base_path,"info.yaml"))
        info={
                "pipelinename":self.pipeline_name,
                "folder":self.pipeline_path,
                "subfolder":self.pipeline_name,
                "created_at":  datetime.datetime.now(),
                "status":"-",
                "jobtime":"",
                "jobtime":"-"
            }
        already_exist=False
        
        for idx,d in enumerate(data):
            if d["pipelinename"]==self.pipeline_name:
                already_exist=True
                exists_idx=idx
        
        if not already_exist:
            data.append(info)
        else:
            data[idx].update(info)
        
            
        yamlio.write_to_yaml(os.path.join(self.base_path,"info.yaml"), data)
        yamlio.write_to_yaml(os.path.join(path_to_yaml,file_name),self.dag)

    def __load__mld_file(self,info:dict)->object:
        """_summary_: Load Mould File with all the injected dependencies 

        Args:
            info (dict): dictinary containing the location of the mould file

        Returns:
            object: returns a python object
        """
        loader_path=os.path.join(self.path,info["root_path"],info["sub_path"],info['filename'])
        return cloudpickle.load(open(loader_path,'rb'))
    
    def load_pipeline(self):
        """_summary_: Load pipeline from specific location
        """
        dag=yamlio.read_yaml(os.path.join(self.pipeline_path,self.pipeline_name,f'{self.pipeline_name}.yaml'))
        self.dag=dag
        
        
        
        
    def _get_input_for_func(self,dag_states:dict,node_dict:dict,out_put_nodes:dict)-> dict:
        """_summary_: get the inputs for each node

        Args:
            dag_states (dict): contains mapped variable  <function_nam>: <arg_name> [the <arg_name> is the argument name as defined in the function] 
            out_put_nodes (dict): contains the previous outputs for the functions that have completed running
            node_info (dict): dictinary containing the location of the mould file and input nodes connected to the given node

        Returns:
            dict: returns a dictionary for  <arg_name>: <prev_output> mapping that can be used in the next node
        """
        # sourcery skip: assign-if-exp, reintroduce-else, swap-if-expression
        
        ##if no args are there
        
        if not dag_states: return dag_states
        input_dict={}
        for func_name_,map_name_ in dag_states.items():
            ##if there are any external arguments  
            if not func_name_.startswith(self._args_tag):
                input_dict[map_name_]=out_put_nodes[func_name_]
            else:
                input_dict[map_name_]=node_dict["args"][map_name_]
        return input_dict
        
    
    def __change_status__(self,status:str,node_name:str,log:str=None):
        """_summary_: Change Node status

        Args:
            status (str): What is the status for the Node
            node_name (str): Name of the Node
            log (str, optional): Any Log files to be added. Defaults to None.
        """
        if status==0:
            self.dag["node_details"][node_name] = {
                "start_time": str(datetime.datetime.now()),
                "log": f"======{self.status_code[status].upper()}======{str(datetime.datetime.now())}\n",
                "status":self.status_code[status]
            }
        elif status in {1, 3}:
            self.dag["node_details"][node_name] = {
                "end_time": str(datetime.datetime.now()),
                "log": f"======{self.status_code[status].upper()}======{str(datetime.datetime.now())}\n",
                "status":self.status_code[status]

            }
            if log!=None:
                self.dag["node_details"][node_name]["log"] += "\n" + log + "======"
            
    def _check_for_job_status(self,jobname,queue_name):
        all_jobs=yamlio.read_yaml(queue_name)
        status=[j["status"] for j in all_jobs if j["pipelinename"]==jobname]
        return status[0] != "Stopped"
    
    
    def bfs(self, graph:dict, entry_node:str,node_info:dict,dag_states:dict): #function for BFS
        """_summary_: Breadth-first search 

        Args:
            graph (dict): contains DAG structure of the nodes "root" is the starting node {root : [nodeA],nodeA :[nodeB, nodeC]}
            entery_node (str): the entry node is the "root" node
            node_info (dict): dictinary containing the location of the mould file and input nodes connected to the given node
            dag_states (dict): ontains mapped variable  <function_nam>: <arg_name> [the <arg_name> is the argument name as defined in the function] 
        """
        
        visited = [entry_node] # List for visited nodes.
        queue = [entry_node]
        
        output_list={}
        while queue:          # Creating loop to visit each node
            m = queue.pop(0)
            for neighbour in graph[m]:
                if neighbour not in visited:
                    try:
                        print("Node-->",neighbour)
                        function_=self.__load__mld_file(node_info[neighbour])
                    
                        output_list[neighbour]=function_(**self._get_input_for_func(dag_states[neighbour],node_info[neighbour],output_list))
                        self.__change_status__(1,neighbour)
                    except Exception as e:
                        print(traceback.format_exc())
                        self.__change_status__(3,neighbour,traceback.format_exc())
                    
                    visited.append(neighbour)
                    queue.append(neighbour)
                    yamlio.write_to_yaml(os.path.join(self.path_pipe,f"{self.pipeline_name}.yaml"),self.dag)
                    
                    
    
    def run(self):
         #Initialize a queue
        dag=self.dag
        self.bfs(dag["graph"],"root",node_info=dag["nodes"],dag_states=dag["args_map"])
        yamlio.write_to_yaml(os.path.join(self.path_pipe,f"{self.pipeline_name}.yaml"),self.dag)

    def run_serialized(self,flag_variable_path,job_name):
         #Initialize a queue
        dag=self.dag
        print("node", self._check_for_job_status(job_name,flag_variable_path))
        if not self._check_for_job_status(job_name,flag_variable_path):
            sys.exit()
            
        #RESET status    
        for node_name in dag["node_details"]:
            self.dag["node_details"][node_name]={"status":self.status_code[2],"start_time":"-","end_time":"-","log":""}
        yamlio.write_to_yaml(os.path.join(self.path_pipe,f"{self.pipeline_name}.yaml"),self.dag)


        # After RUn complete write status code
        self.bfs(dag["graph"],"root",node_info=dag["nodes"],dag_states=dag["args_map"])
        

    def __get_dag__(self):
        return self.dag