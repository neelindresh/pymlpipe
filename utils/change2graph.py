
def search_arch(architecture, node):
    for arch in architecture:
        if arch["layer_name"] == node:
            return arch
    return None

def makegraph(ops,architecture):
    print(ops)
    print(architecture)
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
    