import pickle
import torch
class Deployment:
    def __init__(self,model_path,typeof="non_runtime"):
        
        self.model_path = model_path
        if typeof=="non_runtime":
            self.model=self._load_model(self.model_path)
        elif typeof=="runtime":
            self.model=self._load_model_with_runtime(self.model_path)
            
        
    def _load_model(self,model_name):
        model = torch.jit.load(model_name)
        return model

    def _load_model_with_runtime(self,model_name):
        loaded_trace = torch.jit.load(model_name)
        return loaded_trace
    
    def predict(self,data,dtype):
        status=0
        try:
            if dtype=="float":
                data=torch.from_numpy(data).type(torch.FloatTensor)
                return self.model(data).detach().numpy(),status
            elif dtype=="double":
                data=torch.from_numpy(data).type(torch.DoubleTensor)
                return self.model(data).detach().numpy(),status
            elif dtype=="int":
                data=torch.from_numpy(data).type(torch.IntTensor)
                return self.model(data).detach().numpy(),status
        except Exception as e:
            status=1
            return str(e),status
        
        