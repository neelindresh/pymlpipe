from pymlpipe.utils import yamlio
import os
import time
import pymlpipe.pipeline as pipeline

BASE_DIR=os.getcwd()
queue_store="ML_pipelines"
queue_name="info.yaml"
def execute_from_queue(name,path):
    print("Start execution :"+name)
    ppl=pipeline.Pipeline(name)
    ppl.load_pipeline()
    ppl.run_serialized(flag_variable_path=path,job_name=name)
    print("End execution :")
def change_status(queue,status,job_id=None):
    for idx,job in enumerate(queue) :
        if job["status"]=="Queued" and job_id==None:
            queue[idx]["status"]=status
            return job["pipelinename"],queue
        elif job["status"]=="Started" and job_id==None:
            return None,queue
        elif job["status"]=="Started" and job_id!=None:
            if job["pipelinename"]==job_id:
                queue[idx]["status"]=status
                return job["pipelinename"],queue
            
    return None,queue
            

def start_server():
    
    while True:
        print('-- START--')
        queue=yamlio.read_yaml(os.path.join(BASE_DIR,queue_store,queue_name))
        job_name,queue=change_status(queue,"Started")
        yamlio.write_to_yaml(os.path.join(BASE_DIR,queue_store,queue_name),queue)
        if job_name!=None:
            execute_from_queue(job_name,path=os.path.join(BASE_DIR,queue_store,queue_name))
            job_name,queue=change_status(queue,"Completed",job_name)
            yamlio.write_to_yaml(os.path.join(BASE_DIR,queue_store,queue_name),queue)
        time.sleep(5)
        print('-- END--')