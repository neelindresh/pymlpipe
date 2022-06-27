import flask
import os
from utils import yamlio


app=flask.Flask(__name__)

BASE_DIR=os.getcwd()
MODEL_FOLDER_NAME="modelrun"
MODEL_DIR=os.path.join(BASE_DIR,MODEL_FOLDER_NAME)

EXPERIMENT_FILE="experiment.yaml"

@app.route("/")
def index():
    experiment_lists=yamlio.read_yaml(os.path.join(MODEL_DIR,EXPERIMENT_FILE))
    info={}
    metrics=[]
    for experiment,run_data in experiment_lists.items():
        for run_id in run_data["runs"]:
            run_folder=os.path.join(run_data['experiment_path'],run_id,"info.yaml")
            run_details=yamlio.read_yaml(run_folder)
            info[run_id]=run_details
            if "metrics" in run_details:
                metrics.extend(list(run_details["metrics"].keys()))
            
        
    
    
    return flask.render_template("index.html",
                                 runs=experiment_lists,
                                 run_details=info,
                                 metrics=list(set(metrics))
                                 )

if __name__ == '__main__':
    app.run()