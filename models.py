from typing import List, Optional
from pydantic import BaseModel
from fastapi import Query


class Pipeline(BaseModel):
    pipeline_id : Optional[str] = None
    pipeline_name: str
    task_id: str
    project_id: str
    detection_type  : str
    area : List[List[float]]
    status  : Optional[str]= None

class PipelineModel(BaseModel):
    pipeline: Pipeline
    model_name: Optional[str] = None
    uploaded_geojsons: Optional[List[List[List[float]]]] = None


class Result(BaseModel):
    pipeline_id: str
    gps_bounding_box: List[List[float]]
    extra_info: dict

class Task(BaseModel):
    task_name: str
    project_id: str
    task_id: str
    task_gps: List[float]


class Train(BaseModel):
    training_id : str
    task_id: str
    project_id: str
    detection_type  : str
    uploaded_files : List[List[List[float]]]
    status  : str

class CustomModels(BaseModel):
    training_id: str
    detection_type: str
    path_to_model_weights: str
    status: Optional[str] = None