
#load necessary modules and libraries
from timeit import default_timer
import uuid

from async_timeout import timeout
import uvicorn
import requests
from fastapi import FastAPI
# from fastapi.responses import FileResponse
from typing import List
from models import Pipeline,Result, Task, PipelineModel, CustomModels
from detectionPipeline import detection_pipeline
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Polygon
from rediscli import set_dict_redis
from rq import Queue
from redis import Redis
from config import pipeline_collection, result_collection, odm_link, token, custom_models
from typing import List, Optional
from trainPipeline import train_pipeline, custom_model_inference
import os


tags_metadata = [
    {
        "name": "users",
        "description": "Operations with users. These can be called without any special authentication.",
    },]

description = """
ODM APIs helps you managing monitoring and surveillance using drones. 🚀
"""

# Declaring our FastAPI instance
app = FastAPI(
    title="ODM APIs",
    description=description,
    version="0.0.1",
    contact={
        "name": "Anant Yash Pande",
        "email": "anant.pande@infosys.com",
    },
    openapi_tags=tags_metadata,
)


q = Queue(connection=Redis(host='redis'), default_timeout=72000)
set_dict_redis("pipeline_status", {})
#allow cors
#add cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/tasks-list", response_model=List[Task],tags=["users"])
async def get_task_list():
    """
    API endpoint to return the respective projects and tasks list along with details of the tasks.
    """
    projects_list = requests.get(odm_link + "/api/projects/?jwt="+token).json()
    
    final_response = []

    for i in range(1):
        
        res_tasks = requests.get(odm_link + "/api/projects/1/tasks/?jwt={}".format(token)).json()
        print(res_tasks)
        for j in range(len(res_tasks)):
            tasks = {}
            print(res_tasks[j])
            tasks['name'] = res_tasks[j]['name']
            tasks['project'] = 1
            tasks['task'] = res_tasks[j]['id']

            geojson = requests.get(odm_link + "/api/projects/{}/tasks/{}/download/shots.geojson?jwt={}".format(1,res_tasks[j]['id'],token)).json()

            cord_list = []
            for k in range(len(geojson['features'])):
                # [ , , ]
                cord = geojson['features'][k]['geometry']['coordinates']
                cord_list.append(cord)

            polygon = Polygon(cord_list)
            tasks['gps'] = polygon.bounds


        
            final_response.append(Task(task_name=tasks['name'], project_id=tasks['project'], task_id=tasks['task'], task_gps=tasks['gps']))


    return final_response

# check->query param to body params
@app.get("/pipelines", response_model=List[str],tags=["users"])
async def pipeline_ids(task_id:str,project_id:str):
    """
    API endpoint to return the pipeline ids filtered out by task id and project ids.
    """
    ids = pipeline_collection.find({'task_id':task_id, 'project_id': project_id}) 
    pipline_id_list = []
    for id in ids:
        if id.get('pipeline_id') is not None:
            pipline_id_list.append(id['pipeline_id'])
    return pipline_id_list

# checked
@app.get("/pipeline", response_model=Pipeline,tags=["users"])
async def getpipeline(pipeline_id:str):
    """
    API endpoint to return the details of the respective pipeline.
    """
    pipeline_res = pipeline_collection.find_one({'pipeline_id':pipeline_id})
    return Pipeline(pipeline_id=pipeline_res['pipeline_id'],
                    pipeline_name=pipeline_res['pipeline_name'],
                    task_id=pipeline_res['task_id'],
                    project_id=pipeline_res['project_id'],
                    detection_type=pipeline_res['detection_type'],
                    area=pipeline_res['area'],
                    status=pipeline_res['status'])

def train_custom(detection_info): 
    """
    Function to train a custom object.
    """
    
    if not (os.path.exists(os.path.join(os.getcwd(), 'uploaded_geojsons'))):
        os.mkdir(os.path.join(os.getcwd(), 'uploaded_geojsons'))

    training_info = {}
    training_info['training_id'] = str(uuid.uuid1())
    print("training_id is: ", str(training_info['training_id']))
    training_info['task_id'] = detection_info['task_id']
    training_info['project_id'] =  detection_info['project_id']
    training_info['detection_type'] = detection_info['model_name']
    training_info['list_of_geojson'] = detection_info['uploaded_geojsons']
    training_info['status'] = 'unprocessed'

    document_id = pipeline_collection.insert_one(training_info).inserted_id
    print(f"Training Document with id {document_id} has been created")
    train_pipeline(training_info)

    custom_model_inference(detection_info)


@app.post("/pipeline", response_model=str, tags=["users"])
async def newpipeline(pipeline_obj: PipelineModel):
    """
    API endpoint to initiate a new pipeline once the user provides the input.
    """
    pipeline_obj_dict = pipeline_obj.dict()
    print(pipeline_obj_dict)
    detection_info = {}
    detection_info['pipeline_id'] = str(uuid.uuid1())
    print("pipeline is: ", str(detection_info['pipeline_id']))
    detection_info['pipeline_name'] = pipeline_obj_dict['pipeline']['pipeline_name']
    detection_info['task_id'] = pipeline_obj_dict['pipeline']['task_id']
    detection_info['project_id'] =  pipeline_obj_dict['pipeline']['project_id']
    detection_info['detection_type'] = pipeline_obj_dict['pipeline']['detection_type']
    detection_info['area'] = pipeline_obj_dict['pipeline']['area']
    detection_info['model_name'] = pipeline_obj_dict['model_name']
    detection_info['uploaded_geojsons'] = pipeline_obj_dict['uploaded_geojsons']
    detection_info['status'] = 'unprocessed'

    document_id = pipeline_collection.insert_one(detection_info).inserted_id
    print(f"Pipeline Document with id {document_id} has been created")

    # redis code starts
    pipeline_status = {"status":"unprocessed"}
    pipeline_status[detection_info['pipeline_id']] = {"status":"unprocessed"}
    set_dict_redis(detection_info['pipeline_id'], pipeline_status)

    if(detection_info['detection_type'] == 'Custom'):
        if (detection_info['model_name'] is not None and detection_info['uploaded_geojsons'] is not None):
            q.enqueue(train_custom,detection_info)
        else:
            return "Failed. Required custome model name and geojsons for training..!"

    else:
        results = custom_models.find({})
        result_list = ['Vegetation']
        for res in results: 
            result_list.append(res['detection_type'])
        if detection_info['detection_type'] not in result_list:
            return "Failed. Model not found. Try with Custom option!"
        else:
            if detection_info['detection_type'] == 'Vegetation':
                q.enqueue(detection_pipeline,detection_info)
            else:
                q.enqueue(custom_model_inference,detection_info)
   
    return "SUCCESS"


@app.get("/results", response_model=List[Result],tags=["users"])
async def savedResults(pipeline_id:str):
    """
    API endpoint to return the results obtained after the execution of the pipeline.
    """
    results = result_collection.find({'pipeline_id':pipeline_id})
    result_list = []
    for res in results:
        result_list.append(Result(pipeline_id=res['pipeline_id'],gps_bounding_box=res['gps_bounding_box'],extra_info=res['extra_info']))
    return result_list

@app.get("/custom-models", response_model=List[CustomModels],tags=["users"])
async def custom_models_api():
    """
    API endpoint to return the list of custom models obtained after the execution of the train pipeline.
    """
    results = custom_models.find({})
    result_list = []
    for res in results:
        # if res.get('status') is None:
        #     result_list.append(CustomModels(training_id=res['training_id'], detection_type=res['detection_type'],path_to_model_weights=res['path_to_model_weights']))
        # else:
        result_list.append(CustomModels(training_id=res['training_id'], detection_type=res['detection_type'],path_to_model_weights=res['path_to_model_weights'], status=res['status']))
    return result_list


@app.post("/train-data", response_model=str, tags=["users"])
async def upload(task_id:str, project_id:str, detection_type:str, uploaded_geojsons: List[List[List[float]]]): 
    """
    API endpoint to initiate a new training pipeline once the user provides the input.
    """
    
    if not (os.path.exists(os.path.join(os.getcwd(), 'uploaded_geojsons'))):
        os.mkdir(os.path.join(os.getcwd(), 'uploaded_geojsons'))

    training_info = {}
    training_info['training_id'] = str(uuid.uuid1())
    print("training_id is: ", str(training_info['training_id']))
    training_info['task_id'] = task_id
    training_info['project_id'] =  project_id
    training_info['detection_type'] = detection_type
    training_info['list_of_geojson'] = uploaded_geojsons
    training_info['status'] = 'unprocessed'

    # redis code starts
    pipeline_status = {}
    pipeline_status[training_info['training_id']] = {"status":"unprocessed"}
    set_dict_redis(training_info['training_id'], pipeline_status)

    q.enqueue(train_pipeline,training_info)

    return "SUCCESS"




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
