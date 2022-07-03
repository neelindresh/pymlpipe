import pickle
class Deployment:
    def __init__(self,model_path):
        self.model_path = model_path
        self.model=pickle.load(open(self.model_path,'rb'))
        
    
    def predict(self,data):
        return self.model.predict(data)
        
        