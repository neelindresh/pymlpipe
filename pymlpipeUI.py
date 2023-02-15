import flask
import os
from pymlpipe.utils import yamlio
from pymlpipe.utils import uiutils
from pymlpipe.utils import change2graph
from pymlpipe.utils import database

from flask_api import FlaskAPI
import numpy as np
import json
import uuid
from datetime import datetime

import pandas as pd

app = FlaskAPI(__name__)


BASE_DIR=os.getcwd()
MODEL_FOLDER_NAME="modelrun"
PIPELINE_FOLDER_NAME="ML_pipelines"
MODEL_DIR=os.path.join(BASE_DIR,MODEL_FOLDER_NAME)
PIPELINE_DIR=os.path.join(BASE_DIR,PIPELINE_FOLDER_NAME)

EXPERIMENT_FILE="experiment.yaml"
DEPLOYMENT_FILE="deployment.yaml"
QUEUE_NAME="queue.yaml"

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
        exp_wise_metrics[experiment]=[]
        for run_id in run_data["runs"]:
            print(run_data['experiment_path'],run_id,"info.yaml")
            run_folder=os.path.join(run_data['experiment_path'],run_id,"info.yaml")
            run_details=yamlio.read_yaml(run_folder)
            info[run_id]=run_details
            if 'tags' in run_details:
                tags.extend(run_details["tags"])
            if "metrics" in run_details:
                metrics.extend(list(run_details["metrics"].keys()))
                mm=[i for i in list(run_details["metrics"].keys()) if i not in exp_wise_metrics[experiment]]
                exp_wise_metrics[experiment].extend(mm)
                
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
    
    model_type=""
    metrics_log={}
    metrics_log_plot={}
    graph_dict={}
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
    
    if "metrics_log" in run_details and len(run_details["metrics_log"])>0:
        metrics_log["data"]=run_details["metrics_log"]
        metrics_log["cols"]=list(run_details["metrics_log"][0].keys())
        last_key=None
        for m in metrics_log["data"]:
            for k,v in m.items():
                if k in metrics_log_plot:
                    metrics_log_plot[k].append(v)
                else:
                    metrics_log_plot[k]=[v]
                last_key=k
        
        metrics_log_plot["range"]=list(range(len(metrics_log_plot[last_key])))
    
    
    if "model" in run_details and "model_type" in run_details["model"]:
        model_type=run_details["model"]["model_type"]
    #print(run_details["model"]["model_ops"])
    
    if "model_ops" in run_details["model"]:
        graph_dict=change2graph.makegraph(run_details["model"]["model_ops"],run_details["model"]["model_architecture"])
    XAI=""
    if "XAI" in run_details:
        XAI_temp=run_details["XAI"]
        XAI_feature_map=pd.read_csv(XAI_temp["feature_explainer"])
        XAI_feature_map=XAI_feature_map.round(3)
        print(XAI_temp)
        XAI={
            "table":{
                "columns":XAI_feature_map.columns,
                "rows":XAI_feature_map.values
            },
            "image": flask.Markup(open(XAI_temp["shap"]).read()) if XAI_temp["shap"]!="" else ""
        }
        #print(XAI_feature_map.values)
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
                                 deploy_status=deploy_status,
                                 metrics_log=metrics_log,
                                 metrics_log_plot=metrics_log_plot,
                                 model_type=model_type,
                                 graph_dict=graph_dict,
                                 XAI=XAI
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
    elif model_type=="torch":
        filename=filename+".pt"
    #run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,run_id,'info.yaml'))
    return flask.send_from_directory(os.path.join(MODEL_DIR,experiments,run_id,"models"), filename,as_attachment=True)
    
@app.route("/deployments/<run_id>/")
def deployments(run_id):
    
    experiments,runid=run_id.split("@")
    run_details=yamlio.read_yaml(os.path.join(MODEL_DIR,experiments,runid,'info.yaml'))
    deployed=uiutils.deployment_handler(run_details["model"]["model_path"],
                               run_details["model"]["model_type"],
                               run_details["model"]["model_mode"]) 
    run_hash= str(uuid.uuid3(uuid.NAMESPACE_DNS, run_id)).replace("-", "")[:16]
    if run_hash not in PREDICTORS:
        PREDICTORS[run_hash]=deployed
        ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
        ALL_DEPLOYED_MODELS.append(
            {
                "run_id":runid,
                "experiment_id":experiments,
                "model_path":run_details["model"]["model_path"],
                "model_type":run_details["model"]["model_type"],
                "model_deployment_number": run_hash,
                "model_url":"/predict/"+run_hash,
                "status":'running',
                "model_mode": run_details["model"]["model_mode"]
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
   
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    info_dict={}
    model_type=None
    for model in ALL_DEPLOYED_MODELS:
        if model["model_deployment_number"]==hashno and model["status"]=="running":
            del model['model_path']
            info_dict=model
            model_type=model["model_type"]
            break
    
    if len(info_dict)==0 or hashno not in PREDICTORS:
        return {"info":{
            "error":404,
            "msg":"No such endpoint present"
        }
    }
    if flask.request.method=="POST":
        
        data=flask.request.data
        dtype=None
        if "dtype" in data:
            dtype=data["dtype"]
        predictions,status=PREDICTORS[hashno].predict(np.array(data['data']),dtype)
        if status==1:
            return {
                "deployment no":hashno,
                "error": predictions
            }
        return {
            "deployment no":hashno,
            "predictions":[float(p) for p in predictions]
        }
    if model_type=="scikit-learn":
        return {
            "info":info_dict,
            "request_body":{
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
            }}
    else:
        return {
            "info":info_dict,
            "request_body":{
                    "data": [
                        [ 42.0,
                        120.0,   
                        1.0,   
                        0.0,   
                        0.0,   
                        0.0, 
                        185.7, 
                        133.0,
                        31.57,
                        235.1,
                        149.0,
                        19.98,
                        256.4,
                        78.0,
                        11.54,
                        16.9,
                        6.0,
                        4.56,
                        0.0  
                    ]
                ],
                "dtype": "float"
            }
            
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
            model_type=ALL_DEPLOYED_MODELS[idx]["model_type"]
            
            PREDICTORS[deployment_no]=uiutils.deployment_handler(d["model_path"], model_type, d["model_mode"])
    yamlio.write_to_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE),ALL_DEPLOYED_MODELS)
    
    return {"status":200}
                
                                 
@app.route("/jobs/")
def jobs():
    all_pipelines=yamlio.read_yaml(os.path.join(PIPELINE_DIR,"info.yaml"))
    
    return flask.render_template("jobs.html",
                                 pipeline=all_pipelines
                                 )
    
@app.route("/jobs/run/<runid>")
def runjobs(runid):
    #all_pipelines=yamlio.read_yaml(os.path.join(PIPELINE_DIR,QUEUE_NAME))
    all_pipelines=yamlio.read_yaml(os.path.join(PIPELINE_DIR,"info.yaml"))
    '''
    all_pipelines.append({
        "pipelinename":runid,
        "datetime": datetime.now(),
        "status":"Queued",
        "ops":{}
    })
    '''
    for idx,p in enumerate(all_pipelines):
        if p["pipelinename"]==runid:
            if all_pipelines[idx]["status"]=="Started":
                all_pipelines[idx]["status"]="Stopped"
                all_pipelines[idx]["jobtime"]=datetime.now()
            else:
                all_pipelines[idx]["status"]="Queued"
                all_pipelines[idx]["jobtime"]=datetime.now()
            
            
        
    yamlio.write_to_yaml(os.path.join(PIPELINE_DIR,"info.yaml"),all_pipelines)
    return flask.redirect(flask.url_for("jobs"))

@app.route("/jobs/view/<runid>")
def viewjobs(runid):
    #all_pipelines=yamlio.read_yaml(os.path.join(PIPELINE_DIR,QUEUE_NAME))
    all_pipelines=yamlio.read_yaml(os.path.join(PIPELINE_DIR,runid,runid+".yaml"))
    
    grapg_dict=change2graph.makegraph_pipeline(all_pipelines["graph"],all_pipelines["node_details"])
    nodes_logs={k:all_pipelines["node_details"][k]["log"] for k in all_pipelines["node_details"]}
    #nodes_logs={}
    return flask.render_template("job_view.html",
                                 pipelinename=runid,
                                 grapg_dict=grapg_dict,
                                 nodes=nodes_logs,
                                 initital_node=nodes_logs[list(nodes_logs.keys())[0]]
                                 )
    
    
    
def start_ui(host=None,port=None,debug=False):
    '''Implemet logic for try catch'''
    
    ALL_DEPLOYED_MODELS=yamlio.read_yaml(os.path.join(MODEL_DIR,DEPLOYMENT_FILE))
    for i in ALL_DEPLOYED_MODELS:
        model_type=i["model_type"]
        
        deployed=uiutils.deployment_handler(i["model_path"], model_type, i["model_mode"])
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