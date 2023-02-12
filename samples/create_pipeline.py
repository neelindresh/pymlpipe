from pymlpipe import pipeline
import pandas as pd
ppl=pipeline.Pipeline("PIPELINEV2")


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
def node5(node1_df,node2_df,node5_df,node6_df,path):
    print(node1_df.append(node2_df))
    
n1=ppl.add_node("node1", node1,entry_node=True)
n2=ppl.add_node("node2", node3,node_input=["node1"])
n3=ppl.add_node("node3", node2,node_input=["node1"])
n5=ppl.add_node("node5", node2,node_input=["node1"])
n6=ppl.add_node("node6", node2,node_input=["node1"])
n4=ppl.add_node("node4", node4,node_input=["node2","node3"])
n7=ppl.add_node("node7", node5,node_input=["node5","node6"])


ppl.add_edge(n1, n2)
ppl.add_edge(n1, n3)
ppl.add_edge(n1, n5)
ppl.add_edge(n1, n6)

ppl.add_edge(n2, n4)
ppl.add_edge(n3, n4)
ppl.add_edge(n5, n7)
ppl.add_edge(n6, n7)


#n1>>[n2,n3]
ppl.register()