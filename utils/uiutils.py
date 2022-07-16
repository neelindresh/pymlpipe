
from pymlpipe.utils import _sklearn_prediction,_torch_prediction


def deployment_handler(model_path,model_type,runtime):
    if model_type=="scikit-learn":
        
        deployed=_sklearn_prediction.Deployment(model_path)
    elif model_type=="torch":
        deployed=_torch_prediction.Deployment(model_path,typeof=runtime)
    return deployed