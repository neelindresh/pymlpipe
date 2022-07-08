'''
elements: {
			
                        
					nodes: [
				        { data: { id: 'A' } },
				        { data: { id: 'B' } },
				        { data: { id: 'C' } },
				        { data: { id: 'D' } },
				        { data: { id: 'E' } },
				        { data: { id: 'F' } },
				        { data: { id: 'G' } },
				        { data: { id: 'H' } },
				        { data: { id: 'J' } },
				        { data: { id: 'K' } },
				        { data: { id: 'L' } },
				        { data: { id: 'M' } }
				      
				    ],
				  edges: [
  				      { data: { id: 'e1', source: 'A', target: 'B' } },
  				      { data: { id: 'e2', source: 'A', target: 'C' } },
  				      { data: { id: 'e3', source: 'B', target: 'D' } },
  				      { data: { id: 'e4', source: 'C', target: 'D' } },
  				      { data: { id: 'e5', source: 'C', target: 'E' } },
  				      { data: { id: 'e6', source: 'C', target: 'F' } },
  				      { data: { id: 'e7', source: 'D', target: 'G' } },
  				      { data: { id: 'e8', source: 'D', target: 'H' } },
  				      { data: { id: 'e9', source: 'E', target: 'H' } },
  				      { data: { id: 'e10', source: 'E', target: 'J' } },
  				      { data: { id: 'e11', source: 'F', target: 'J' } },
  				      { data: { id: 'e12', source: 'F', target: 'K' } },
  				      { data: { id: 'e13', source: 'G', target: 'L' } },
  				      { data: { id: 'e14', source: 'H', target: 'L' } },
  				      { data: { id: 'e15', source: 'H', target: 'M' } },
  				      { data: { id: 'e16', source: 'J', target: 'M' } }
				    ]
				},
'''

def search_arch(architecture,node):
    for arch in architecture:
        if arch["layer_name"] ==node:
            return arch
    return None

def makegraph(ops,architecture):
    
    graph_dict={"nodes":[],"edges":[]}
    for op in ops:
        #prev=ops[op]['name'] if ops[op]['prev']=="" else ops[op]['prev']
        #next_pt=ops[op]['name'] if ops[op]['next']=="" else ops[op]['next']
        arch_details=search_arch(architecture, ops[op]['name'])
        if arch_details!=None:
            graph_dict["nodes"].append({'data':{ 'id': ops[op]['name'] ,
                                            "label":op,
                                            "type":arch_details["layer_type"] ,
                                            "details":[k+"="+str(v) for k,v in arch_details["params"].items()]
                                            } })
        else:
            graph_dict["nodes"].append({'data':{ 'id': ops[op]['name'] ,
                                            "label":op,
                                            "type": ops[op]['name'],
                                            "details":[ops[op]['op']]
                                            } })
        if ops[op]['next']!="":
            graph_dict["edges"].append({ 'data': { 'id': op, 'source': ops[op]['name'], 'target': ops[op]['next']} })
    return graph_dict
    