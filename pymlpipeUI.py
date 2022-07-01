import flask
import os
from pymlpipe.utils import yamlio


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
            
    exp_names=list(experiment_lists.keys())
    
    
    return flask.render_template("index.html",
                                 runs=experiment_lists,
                                 run_details=info,
                                 metrics=list(set(metrics)),
                                 current_experiment=exp_names
                                 )
@app.route("/run/<run_id>/")
def runpage(run_id):
    experiments,run_id=run_id.split("@")
    experiment_lists=yamlio.read_yaml(os.path.join(MODEL_DIR,EXPERIMENT_FILE))
    run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,run_id,'info.yaml'))
    expertiment_details={
        "RUN_ID":run_id,
        "EXPERIMENT NAME":experiments,
        "EXECUTION DATE TIME":run_details["execution_time"]
        }
    if 'tags' in run_details:
        expertiment_details["TAGS"]=run_details['tags']
    else:
        expertiment_details["TAGS"]="-"
    if 'version' in run_details:
        expertiment_details["VERSION"]=run_details['version']
    else:
        expertiment_details["VERSION"]="-"
    
    return flask.render_template('run.html',
                                 run_id=run_id,
                                 experiments=experiments,
                                 expertiment_details=expertiment_details,
                                 artifact_details=run_details["artifact"],
                                 metrics_details=run_details["metrics"],
                                 model_details=run_details["model"],
                                 param_details=run_details["params"],
                                 
                                 )
@app.route("/download_artifact/<uid>")
def download_artifact(uid):
    experiments,run_id,filename=uid.split("@")
    #run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,run_id,'info.yaml'))
    return flask.send_from_directory(os.path.join(MODEL_DIR,experiments,run_id,"artifacts"), filename,as_attachment=True)

@app.route("/download_model/<uid>")
def download_model(uid):
    experiments,run_id,filename,model_type=uid.split("@")
    if model_type=="scikit-learn":
        filename=filename+".pkl"
    #run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,run_id,'info.yaml'))
    return flask.send_from_directory(os.path.join(MODEL_DIR,experiments,run_id,"models"), filename,as_attachment=True)
    
@app.route("/deployments/<run_id>/")
def deployments(run_id):
    experiments,run_id=run_id.split("@")
    return {}


def start_ui(host=None,port=None,debug=False):
    '''Implemet logic for try catch'''
    if host==None and port==None:
        app.run(debug=debug)
    elif host==None:
        app.run(port=port,debug=debug)
    elif port==None:
        app.run(host=host,debug=debug)
    else:
        app.run(host=host,port=port,debug=debug)
        
        
    
    

if __name__ == '__main__':
    app.run()