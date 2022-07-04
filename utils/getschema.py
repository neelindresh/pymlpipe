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
            'min':flaot(round(data[col].min(),2)),
            'max':flaot(round(data[col].max(),2)),
            'std':flaot(round(data[col].std(),2)),
            "variance":flaot(round(data[col].var(),2)),
            "mean":flaot(round(data[col].mean(),2)),
            "median":flaot(round(data[col].median(),2)),
            "data type":str(data[col].dtype),
            "unique_values":int(len(data[col].unique())),
            "25th percentile":str(round(data[col].quantile(0.25),2)),
            "50% percentile":str(round(data[col].quantile(0.5),2)),
            "75% percentile":str(round(data[col].quantile(0.75),2)),
        }
        if len(details)==0:
            details=list(schema[col].keys())
    #print("-------->",schema)
    return schema,details