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
            'min':float("{0:.4f}".format(data[col].min())),
            'max':float("{0:.4f}".format(data[col].max())),
            'std':float("{0:.4f}".format(data[col].std())),
            "variance":float("{0:.4f}".format(data[col].var())),
            "mean":float("{0:.4f}".format(data[col].mean())),
            "median":float("{0:.4f}".format(data[col].median())),
            "data type":str(data[col].dtype),
            "unique_values":int(len(data[col].unique())),
            "25th percentile":float("{0:.4f}".format(data[col].quantile(0.25))),
            "50% percentile":float("{0:.4f}".format(data[col].quantile(0.5))),
            "75% percentile":float("{0:.4f}".format(data[col].quantile(0.75))),
        }
        if len(details)==0:
            details=list(schema[col].keys())
    #print("-------->",schema)
    return schema,details