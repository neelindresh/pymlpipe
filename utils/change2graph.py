
def search_arch(architecture, node):
    for arch in architecture:
        if arch["layer_name"] == node:
            return arch
    return None

def makegraph(ops,architecture):
    #print(ops)
    #print(architecture)
    graph_dict={"nodes":[],"edges":[]}
    for op in ops:
        #prev=ops[op]['name'] if ops[op]['prev']=="" else ops[op]['prev']
        #next_pt=ops[op]['name'] if ops[op]['next']=="" else ops[op]['next']
        arch_details=search_arch(architecture, ops[op]['name'])
        if arch_details!=None:
            graph_dict["nodes"].append({'data':{ 'id': ops[op]['name'] ,
                                            "label":op+"_"+ops[op]['name']  ,
                                            "type":arch_details["layer_type"] ,
                                            "details":[k+"="+str(v) for k,v in arch_details["params"].items()]
                                            } })
        else:
            graph_dict["nodes"].append({'data':{ 'id': ops[op]['name'] ,
                                            "label":op+"_"+ops[op]['name'],
                                            "type": ops[op]['name'],
                                            "details":[ops[op]['op']]   
                                            } })
        if ops[op]['next']!="":
            graph_dict["edges"].append({ 'data': { 'id': op, 'source': ops[op]['name'], 'target': ops[op]['next']} })
    return graph_dict
    
    
'''
def makegraph_pipeline(edges,sequence,node_details):
    #print(node_details)
    graph_dict={"nodes":[],"edges":[]}
    color={"Queued":"#828282","Completed":"#80ff80","Failed":"#fc3d03","Started":"#ffff33"}
    for op in sequence:
        #print(op)
        graph_dict["nodes"].append({'data':{ 'id': op ,
                                            "label":op ,
                                            "color":color[node_details[op]["status"]]
                                            } })
        
        for edge in edges:
            if edge["src"]==op:
                graph_dict["edges"].append({'data':{ 'id': edge["src"]+ edge["target"], 'source': edge["src"], 'target': edge["target"]} })
                
    return graph_dict
'''
def makegraph_pipeline(graph:dict,node_details:dict):
    """_summary_: Make graph format for Web Visualization

    Args:
        graph (dict): Contains the data structure for node -edge connection
        node_details (dict): Contains status and log history of nodes
    Returns:
        dict: Returns a dictionary with web format
    """
    color={"Queued":"#828282","Completed":"#80ff80","Failed":"#fc3d03","Started":"#ffff33"}
    entry_node="root"
    graph_dict={"nodes":[],"edges":[]}
    _args_tag="args@"
    for op in graph:
        if op.startswith(_args_tag):
            op=op.strip(_args_tag)
        if entry_node==op:
            graph_dict["nodes"].append({'data':{ 'id': op ,
                                            "label":op ,
                                            "color":color["Completed"]
                                            } })
        else:
            graph_dict["nodes"].append({'data':{ 'id': op ,
                                            "label":op ,
                                            "color":color[node_details[op]["status"]]
                                            } })
        for edge in graph[op]:
            graph_dict["edges"].append({'data':{ 'id': op+ edge, 'source': op, 'target': edge} })
    return graph_dict