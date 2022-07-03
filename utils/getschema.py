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
            'min':round(data[col].min(),2),
            'max':round(data[col].max(),2),
            'std':round(data[col].std(),2),
            "variance":round(data[col].var(),2),
            "mean":round(data[col].mean(),2),
            "median":round(data[col].median(),2),
            "data type":str(data[col].dtype),
            "unique_values":len(data[col].unique()),
            "25th percentile":str(round(data[col].quantile(0.25),2)),
            "50% percentile":str(round(data[col].quantile(0.5),2)),
            "75% percentile":str(round(data[col].quantile(0.75),2)),
        }
        if len(details)==0:
            details=list(schema[col].keys())
    #print("-------->",schema)
    return schema,details