import flask
import os
from pymlpipe.utils import yamlio
from pymlpipe.utils import _sklearn_prediction
from flask_api import FlaskAPI
import numpy as np
import json
import uuid
#app=flask.Flask(__name__)

app = FlaskAPI(__name__)

BASE_DIR=os.getcwd()
MODEL_FOLDER_NAME="modelrun"
MODEL_DIR=os.path.join(BASE_DIR,MODEL_FOLDER_NAME)

EXPERIMENT_FILE="experiment.yaml"
DEPLOYMENT_FILE="deployment.yaml"
#ALL_DEPLOYED_MODELS=[]
PREDICTORS={}
app.secret_key="PYMLPIPE_SEC_KEY"


@app.route("/")
def index():
    '''
    if "status" in flask.request.args:
        
        if flask.request.args["status"]=="501":
            deploy_status=False
            
    '''       
    metric_filters={}
    tag_filters=[]
    if len(flask.request.args):
        if "metrics" in flask.request.args:
            metric_filters[flask.request.args['metrics']]=flask.request.args["metricsfilter"]
        elif "tags" in flask.request.args:
            tag_filters=flask.request.args["tags"].split(",")
    
    experiment_lists=yamlio.read_yaml(os.path.join(MODEL_DIR,EXPERIMENT_FILE))
    if len(experiment_lists)==0:
        return flask.render_template("index.html",
                                    runs=[],
                                    run_details={},
                                    metrics=[],
                                    current_experiment=None
                                     )
    info={}
    metrics=[]
    exp_wise_metrics={}
    tags=[]
    error=""
    for experiment,run_data in experiment_lists.items():
        for run_id in run_data["runs"]:
            print(run_data['experiment_path'],run_id,"info.yaml")
            run_folder=os.path.join(run_data['experiment_path'],run_id,"info.yaml")
            run_details=yamlio.read_yaml(run_folder)
            info[run_id]=run_details
            if 'tags' in run_details:
                tags.extend(run_details["tags"])
            if "metrics" in run_details:
                metrics.extend(list(run_details["metrics"].keys()))
                exp_wise_metrics[experiment]=list(run_details["metrics"].keys())
                
    #filter emmpty runs:            
    info={run:info[run] for run in info if len(info[run])>0}
    
    
    if len(metric_filters)>0:
        newinfo={}
        for run_id,details in info.items():
            
            for mfilter in metric_filters:
                if mfilter in details["metrics"]:
                    fv=details["metrics"][mfilter]
                    try:
                        if eval(str(fv)+metric_filters[mfilter]):
                            newinfo[run_id]=details
                    except Exception as e:
                        error=e
                else:
                    newinfo[run_id]=details
        info=newinfo
    elif len(tag_filters)>0:
        newinfo={}
        for run_id,details in info.items():
            if len(set(tag_filters).intersection(set(details["tags"])))>0:
                newinfo[run_id]=details
        info=newinfo
    
    exp_names=list(experiment_lists.keys())
    
    
    return flask.render_template("index.html",
                                 runs=experiment_lists,
                                 run_details=info,
                                 metrics=list(set(metrics)),
                                 current_experiment=exp_names,
                                 tags=list(set(tags)),
                                 exp_wise_metrics=exp_wise_metrics,
                                 error=error
                                 )
@app.route("/run/<run_id>/")
def runpage(run_id):
    deploy_status=True
    if "status" in flask.request.args:
        
        if flask.request.args["status"]=="501":
            deploy_status=False
    
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
                                 schema_details=run_details["artifact_schema"],
                                 is_deployed=True if "model_path" in run_details["model"] else False,
                                 deploy_status=deploy_status
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
    
    experiments,runid=run_id.split("@")
    run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,runid,'info.yaml'))
    deployed=_sklearn_prediction.Deployment(run_details["model"]["model_path"])
    print(uuid.NAMESPACE_DNS)
    run_hash= str(uuid.uuid3(uuid.NAMESPACE_DNS, run_id)).replace("-", "")[:16]
    if run_hash not in PREDICTORS:
        PREDICTORS[run_hash]=deployed
        ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
        ALL_DEPLOYED_MODELS.append(
            {
                "run_id":runid,
                "experiment_id":experiments,
                "model_path":run_details["model"]["model_path"],
                "model_deployment_number": run_hash,
                "model_url":"/predict/"+run_hash,
                "status":'running'
            }    
        )
        yamlio.write_to_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE),ALL_DEPLOYED_MODELS)
        return flask.redirect(flask.url_for("show_deployments"))
    return flask.redirect("/run/"+run_id+"?status=501")

@app.route("/show_deployments/")
def show_deployments():
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    return flask.render_template('deployments.html',
                                 ALL_DEPLOYED_MODELS=ALL_DEPLOYED_MODELS
                                 )
    
    
@app.route("/predict/<hashno>",methods=["GET","POST"])
def predict(hashno):
    print(PREDICTORS)
    if flask.request.method=="POST":
        #data=flask.request.form['random_data']
        data=flask.request.data
        predictions=PREDICTORS[hashno].predict(np.array(data['data']))
        
        return {
            "deployment no":hashno,
            "predictions":[int(p) for p in predictions]
        }
    return {
        "data":[
            [
                5.6,
                3.0,
                4.5,
                1.5
            ],
            [
                5.6,
                3.0,
                4.5,
                1.5
            ]
        ]
    }
@app.route("/deployment/stop/<deployment_no>",methods=["GET"])                
def stop_deployment(deployment_no):
    global PREDICTORS
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    for idx,d in enumerate(ALL_DEPLOYED_MODELS):
        
        if d['model_deployment_number']==deployment_no:
            print("here here")
            ALL_DEPLOYED_MODELS[idx]['status']="stopped"
    yamlio.write_to_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE),ALL_DEPLOYED_MODELS)
    PREDICTORS={i:j for i,j in PREDICTORS.items() if i!=deployment_no}
    return {"status":200}


@app.route("/deployment/start/<deployment_no>",methods=["GET"])                
def start_deployment(deployment_no):
    global PREDICTORS
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    for idx,d in enumerate(ALL_DEPLOYED_MODELS):
       
        if d['model_deployment_number']==deployment_no:
            
            ALL_DEPLOYED_MODELS[idx]['status']="running"
            PREDICTORS[deployment_no]=_sklearn_prediction.Deployment(d["model_path"])
    yamlio.write_to_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE),ALL_DEPLOYED_MODELS)
    
    return {"status":200}
                
                                 


def start_ui(host=None,port=None,debug=False):
    '''Implemet logic for try catch'''
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    for i in ALL_DEPLOYED_MODELS:
        deployed=_sklearn_prediction.Deployment(i["model_path"])
        PREDICTORS[i['model_deployment_number']]=deployed
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