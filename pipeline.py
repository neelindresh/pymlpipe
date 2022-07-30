import dill
import cloudpickle
from pymlpipe.utils import database,yamlio
import os

class Node:
    def __init__(self,name, func,path):
        self.name=name
        self.func=func
        self.path=path
        self.save(name, func)
        
    def save(self,name,func):
        #mainify(func)
        #dill.dump(func, open(name+".pkl", "wb"))
        
        self.filename=os.path.join(self.path,name+".pkl")
        cloudpickle.dump(func, open(self.filename, "wb"))
        
    
class Pipeline:
    def __init__(self,name):
        path=os.getcwd()
        self.PIPELINE_FOLDER="ML_pipelines"
        database.create_folder(path)
        self.name = name
        self.path_pipe=database.create_folder(path,self.PIPELINE_FOLDER)
        self.path_pipe=database.create_folder(self.path_pipe,self.name)
        
        self.dag={"sequence":[],"nodes":{},"edges":[],"node_order":{}}
        self.sequence=[]
        self.node_order={}
        self.is_entry_node=False
        
    def _make_edges(self,node,edges):
        edge_list=[]
        for edge in edges:
            edge_list.append({"src":edge,"target":node})
        return edge_list
    
    
    def add_node(self,name,func,node_input=None,entry_node=False,args=None):
        if name in self.sequence:
            raise ValueError(f"Node Name {name} already exists! Please provide different Name")
        self.sequence.append(name)
        node=Node(name,func,self.path_pipe)
        self.dag["nodes"][name]={"path":node.filename,"entry":entry_node,"args":args}
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
        if not self.is_entry_node:
            raise ValueError("Entry Node is not defined!!! Please 'entry_node'=True for the starting node")
        self.dag["sequence"]=self.sequence
        self.dag["node_order"]=self.node_order
        #self.dag["graph"]=self._order_node()
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
            
        #print("input-->",inp)
        return inp
        
    
    def bfs(self,graph,entry_node,_prev_outputs,_functions,_node_order,functions_args):
        visited = [entry_node] # List to keep track of visited nodes.
        queue = [entry_node]     #Initialize a queue
        while queue:
            s = queue.pop(0) 
            
            if s in graph:
                for neighbour in graph[s]:
                    if neighbour not in visited:
                        func=_functions[neighbour]
                        
                        #print(self._make_previous_output(_prev_outputs,_node_order[neighbour]))
                        _prev_outputs[neighbour]=func(*self._make_previous_output(_prev_outputs,_node_order[neighbour],functions_args[neighbour]))
                        #func()
                        visited.append(neighbour)
                        queue.append(neighbour)
        return _prev_outputs
    
    
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
            functions[node]=self.load(self.dag["nodes"][node]["path"])
            functions_args[node]=self.dag["nodes"][node]["args"]
        graph=self._create_graph(self.dag["edges"])
        
        for node in entrynode:
            func=functions[node]
            output_nodes[node]=func(*args,**kwargs)
            output_nodes=self.bfs(graph,node,output_nodes,functions,self.dag["node_order"],functions_args)
        return output_nodes
            
            
            
            
        
        
                
        
            
        
            
        
        

