import pickle
class Deployment:
    def __init__(self,model_path):
        self.model_path = model_path
        self.model=pickle.load(open(self.model_path,'rb'))
        
    
    def predict(self,data,dtype):
        status=0
        try:
            return self.model.predict(data),status
        except Exception as e:
            status=1
            return str(e),status
        
        