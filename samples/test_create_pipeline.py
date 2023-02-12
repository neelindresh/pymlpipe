#from pymlpipe import pipeline
from pymlpipe import pipeline
import pandas as pd

#ppl=pipeline.Pipeline("PIPELINEV2")

ppl=pipeline.PipeLine("PIPELINEV2")
def node1():
    path="train.csv"
    df=pd.read_csv(path)
    return df
def node2(df):
    stats=df.describe()
    stats.columns=[col+"_node2" for col in stats.columns]
    return stats
def node3(df):
    stats=df.describe()
    stats.columns=[col+"_node3" for col in stats.columns]
    return stats
    
def node4(node1_df,node2_df):
    print(node1_df.append(node2_df))
def node5(node1_df,node2_df,node3df):
    print(node1_df.append(node2_df))
    
ppl.add_node("node1", node1,entry_node=True)
ppl.add_node("node2", node3,input_nodes=["node1"])
ppl.add_node("node3", node2,input_nodes=["node1"])
ppl.add_node("node5", node2,input_nodes=["node1"])
ppl.add_node("node6", node2,input_nodes=["node1"])
ppl.add_node("node4", node4,input_nodes=["node2","node3"])
ppl.add_node("node7", node5,input_nodes=["node5","node6"])




#n1>>[n2,n3]
ppl.register_dag()
ppl.run()