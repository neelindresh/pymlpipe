import pandas as pd
def schema_(data):
    """_summary_: Generate schema object for a dataframe

    Args:
        data (Pandas DataFrame): Pandas Artifact

    Returns:
        dict: with column schema
    """
    schema={}
    details=[]
    for col in data:
        schema[col]={
            'min':float(data[col].min()),
            'max':float(data[col].max()),
            'std':float(data[col].std()),
            "variance":float(data[col].var()),
            "mean":float(data[col].mean()),
            "median":float(data[col].median()),
            "data type":str(data[col].dtype),
            "unique_values":int(len(data[col].unique())),
            "25th percentile":str(data[col].quantile(0.25)),
            "50% percentile":str(data[col].quantile(0.5)),
            "75% percentile":str(data[col].quantile(0.75)),
        }
        if len(details)==0:
            details=list(schema[col].keys())
    #print("-------->",schema)
    return schema,details