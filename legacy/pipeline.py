import dill
import cloudpickle
from pymlpipe.utils import database,yamlio
import os
import datetime
import traceback,sys
#checking

class Node:
    def __init__(self,name, func,path):
        self.name=name
        self.func=func
        self.path=path
        self.save(name, func)
        
    def save(self,name,func):
        #mainify(func)
        #dill.dump(func, open(name+".pkl", "wb"))
        
        self.filename=os.path.join(self.path,name+".mld")
        cloudpickle.dump(func, open(self.filename, "wb"))
        
    
class Pipeline:
    def __init__(self,name):
        path=os.getcwd()
        self.PIPELINE_FOLDER="ML_pipelines"
        database.create_folder(path)
        self.name = name
        self.base_path=database.create_folder(path,self.PIPELINE_FOLDER)
        self.path_pipe=database.create_folder(self.base_path,self.name)
        
        self.dag={"sequence":[],"nodes":{},"edges":[],"node_order":{},"node_details":{}}
        self.sequence=[]
        self.node_order={}
        self.is_entry_node=False
        
    def _make_edges(self,node,edges):
        """Create the DAG edges for the nodes, in a src--> trg format

        Args:
            node (_type_): _description_
            edges (_type_): _description_

        Returns:
            _type_: _description_
        """
        edge_list=[]
        for edge in edges:
            edge_list.append({"src":edge,"target":node})
        return edge_list
    
    
    def add_node(self,name,func,node_input=None,entry_node=False,args=None):
        if name in self.sequence:
            raise ValueError(f"Node Name {name} already exists! Please provide different Name")
        self.sequence.append(name)
        node=Node(name,func,self.path_pipe)
        self.dag["nodes"][name]={"path":node.filename,"mould_file":node.name+".mld","entry":entry_node,"args":args,"folder":self.PIPELINE_FOLDER,"subfolder":self.name,}
        self.dag["node_details"][name]={"status":"Queued","start_time":"-","end_time":"-","log":""}
        if entry_node:
            self.is_entry_node=True
        if node_input!=None:
            self.node_order[name]=node_input
        return node
        
    def load(self,name):  
        #return dill.load(open(name+".pkl", "rb"))
        #return cloudpickle.load(open(os.path.join(self.path_pipe,self.name+".yaml"), "rb"))
        return cloudpickle.load(open(name,'rb'))
    
    
    def add_edge(self,node_1,node_2):
        if not isinstance(node_1,Node) and isinstance(node_2,Node):
            raise TypeError("node_1 or node_2 is not type Node")
        self.dag["edges"].append({"src":node_1.name,"target":node_2.name})
        
    
    def register(self):
        already_exist=False
        exists_idx=None
        if not self.is_entry_node:
            raise ValueError("Entry Node is not defined!!! Please 'entry_node'=True for the starting node")
        self.dag["sequence"]=self.sequence
        self.dag["node_order"]=self.node_order
        #self.dag["graph"]=
        graph={}
        '''
        for seq in self.sequence:
            if seq in graph:
                graph[seq].append([{"edges":i,"status":None} for i in self.dag["edges"] if i["src"]==seq])
            else:
                graph[seq]=[{"edges":i,"status":None} for i in self.dag["edges"] if i["src"]==seq]
        print(graph)
        '''
        data=yamlio.read_yaml(os.path.join(self.base_path,"info.yaml"))
        
        for idx,d in enumerate(data):
            if d["pipelinename"]==self.name:
                already_exist=True
                exists_idx=idx
        if not already_exist:
            data.append({
                "pipelinename":self.name,
                "path":self.path_pipe,
                "folder":self.PIPELINE_FOLDER,
                "subfolder":self.name,
                "created_at":  datetime.datetime.now(),
                "status":"-",
                "jobtime":"",
                "jobtime":"-"
            })
        else:
            data[idx].update({
                "pipelinename":self.name,
                "path":self.path_pipe,
                "folder":self.PIPELINE_FOLDER,
                "subfolder":self.name,
                "created_at":  datetime.datetime.now(),
                "status":"-",
                "jobtime":"",
                "jobtime":"-"
            })
            
        yamlio.write_to_yaml(os.path.join(self.base_path,"info.yaml"), data)
        yamlio.write_to_yaml(os.path.join(self.path_pipe,self.name+".yaml"), self.dag)
    
    def load_pipeline(self):
        self.dag=yamlio.read_yaml(os.path.join(self.path_pipe,self.name+".yaml"))
    
    def _find_next_node(self,node_name):
        return self.dag["graph"][node_name]
    
    def _create_graph(self,edges):
        graph={}
        for edge in edges:
            if edge["src"] in graph:
                graph[edge["src"]].append(edge["target"])
            else:
                graph[edge["src"]]=[edge["target"]]
        return graph
    def _make_previous_output(self,_prev_outputs,neighbor,functions_args):
        inp=[]
        #print(functions_args)
        for n in neighbor:
            #when output --> tuple,list
            if isinstance(_prev_outputs[n], tuple) or isinstance(_prev_outputs[n], list):
                inp.extend(list(_prev_outputs[n]))
               
                if functions_args!=None:
                    inp.extend(functions_args)
            #make output --> dict
            else: # or isinstance(_prev_outputs[n], str) or isinstance(_prev_outputs[n], int) or isinstance(_prev_outputs[n], float):
                inp.append(_prev_outputs[n])
                
                if functions_args!=None:
                    inp.extend(functions_args)
            #make output --> str,float,int
            
        print("input-->",inp)
        return inp
        
    def _change_status(self,node,status,info=None):
        dag=yamlio.read_yaml(os.path.join(self.path_pipe,self.name+".yaml"))
        if status=="Started":
            
            dag["node_details"][node]["status"]=status
            
            dag["node_details"][node]["start_time"]=str(datetime.datetime.now())
            dag["node_details"][node]["log"]="======"+status.upper()+"======"+str(datetime.datetime.now())+"\n"
        elif status=="Completed" or status=="Failed":
            dag["node_details"][node]["status"]=status
            dag["node_details"][node]["end_time"]=str(datetime.datetime.now())
            dag["node_details"][node]["log"]+="======"+status.upper()+"======"+str(datetime.datetime.now())+"\n"
            if info!=None:
                dag["node_details"][node]["log"]+="\n"+str(info)+"======"
                
                
        dag=yamlio.write_to_yaml(os.path.join(self.path_pipe,self.name+".yaml"),dag)
        
        
        
    def bfs(self,graph,entry_node,_prev_outputs,_functions,_node_order,functions_args,job_name,flag_variable_path):
        visited = [entry_node] # List to keep track of visited nodes.
        queue = [entry_node]     #Initialize a queue
        while queue:
            s = queue.pop(0) 
            
            if s in graph:
                for neighbour in graph[s]:
                    if neighbour not in visited:
                        func=_functions[neighbour]
                        self._change_status(neighbour,"Started")
                        if not self._check_for_job_status(job_name,flag_variable_path): sys.exit() 
                        #print(self._make_previous_output(_prev_outputs,_node_order[neighbour]))
                        print(_prev_outputs)
                        try:
                            _prev_outputs[neighbour]=func(*self._make_previous_output(_prev_outputs,_node_order[neighbour],functions_args[neighbour]))
                            
                            self._change_status(neighbour,"Completed")
                            
                        #func()
                        except Exception as e:
                            #print(neighbour)
                            print(traceback.format_exc())
                            #raceback.print_exception(*sys.exc_info())
                            self._change_status(neighbour,"Failed",info=traceback.format_exc())
                        
                        visited.append(neighbour)
                        queue.append(neighbour)
        return _prev_outputs
    
    def _get_path(self,base_path,folder):
        return os.path.join(base_path,folder)
    def _check_for_job_status(self,jobname,queue_name):
        all_jobs=yamlio.read_yaml(queue_name)
        status=[j["status"] for j in all_jobs if j["pipelinename"]==jobname]
        return False if status[0]=="Stopped" else True
    
    def run(self,*args,**kwargs):
        if len(self.dag["sequence"])==0:
            raise ValueError("Error!!! No Dag Provided!!!!")
        #if not self.is_entry_node:
        #    raise ValueError("Error!!! Entry Node Not defined please provide and entry node with entry_node=True!!!!")
        entrynode=[]
        functions={}
        output_nodes={}
        functions_args={}
        for node in self.dag["nodes"]:
            #print(node,self.dag["nodes"][node])
            if self.dag["nodes"][node]["entry"]:
                entrynode.append(node)
            functions[node]=self.load(self._get_path(self.path_pipe, self.dag["nodes"][node]["mould_file"]))#self.dag["nodes"][node]["path"])
            functions_args[node]=self.dag["nodes"][node]["args"]
        graph=self._create_graph(self.dag["edges"])
        self.dag["node_details"]={node:{"status":"Queued","start_time":"-","end_time":"-","log":""} for node in self.dag["node_details"]}
        yamlio.write_to_yaml(os.path.join(self.path_pipe,self.name+".yaml"), self.dag)
        for node in entrynode:
            func=functions[node]
            self._change_status(node,"Started")
            try:
                print("node", node)
                output_nodes[node]=func(*args,**kwargs)
                self._change_status(node,"Completed")
            except Exception as e:
                print(node)
                
                traceback.print_exception(*sys.exc_info())
                self._change_status(node,"Failed")
            output_nodes=self.bfs(graph,node,output_nodes,functions,self.dag["node_order"],functions_args)
        return output_nodes
    
    
    
    
    def run_serialized(self,flag_variable_path,job_name,*args,**kwargs):
        if len(self.dag["sequence"])==0:
            raise ValueError("Error!!! No Dag Provided!!!!")
        #if not self.is_entry_node:
        #    raise ValueError("Error!!! Entry Node Not defined please provide and entry node with entry_node=True!!!!")
        entrynode=[]
        functions={}
        output_nodes={}
        functions_args={}
        for node in self.dag["nodes"]:
            #print(node,self.dag["nodes"][node])
            if self.dag["nodes"][node]["entry"]:
                entrynode.append(node)
            functions[node]=self.load(self._get_path(self.path_pipe, self.dag["nodes"][node]["mould_file"]))#self.dag["nodes"][node]["path"])
            functions_args[node]=self.dag["nodes"][node]["args"]
        graph=self._create_graph(self.dag["edges"])
        self.dag["node_details"]={node:{"status":"Queued","start_time":"-","end_time":"-","log":""} for node in self.dag["node_details"]}
        yamlio.write_to_yaml(os.path.join(self.path_pipe,self.name+".yaml"), self.dag)
        for node in entrynode:
            func=functions[node]
            self._change_status(node,"Started")
            try:
                print("node", node,self._check_for_job_status(job_name,flag_variable_path))
                if not self._check_for_job_status(job_name,flag_variable_path):
                    sys.exit()
                output_nodes[node]=func(*args,**kwargs)
                self._change_status(node,"Completed")
            except Exception as e:
       
                
                traceback.print_exception(*sys.exc_info())
                self._change_status(node,"Failed")
            output_nodes=self.bfs(graph,node,output_nodes,functions,self.dag["node_order"],functions_args,job_name,flag_variable_path)
        return output_nodes
    
   
            
        
            
            
            
        
        
                
        
            
        
            
        
        

