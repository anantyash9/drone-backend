#load the modules
import os
from models import Result
from rediscli import set_dict_redis, get_dict_redis
from tiles import Tile
from utils import convert_polygon_to_tiles, tree_detection, convert_list, get_detections_coordinates
from config import pipeline_collection, result_collection, odm_link, token, path_to_model_wts


def detection_pipeline(detection_info):
  
  polygon_area = detection_info['area']
  project_id =  detection_info['project_id']
  task_id = detection_info['task_id']
  pipeline_id = detection_info['pipeline_id'] 

  pipeline_status = get_dict_redis(pipeline_id)
  pipeline_status[pipeline_id] = {"status":"processing"}
  set_dict_redis(pipeline_id,pipeline_status)

  pipeline_collection.update_one({"pipeline_id": pipeline_id}, {'$set': {'status': 'processing'}})

  
  # step 1
  server_url = odm_link+'api/projects/'+ project_id + '/tasks/' + task_id + '/orthophoto/tiles/{z}/{x}/{y}.png?jwt=' + token 
  
  #change path as per location
  images_loc="./images" 
  tiles_folder = 'zoom_level_19'
  zoom_level = 19

  polygon_bounds = convert_polygon_to_tiles(polygon_area, server_url, images_loc, zoom_level, tiles_folder)

  # step 2
  path_to_tile_images = os.path.join(images_loc, tiles_folder)

  #change path as per location
  path_to_save_detection = 'detection_images_19/'


  tree_detection(path_to_tile_images, path_to_save_detection, path_to_model_wts)

  response = {}
  # for loop
  for images in os.listdir(path_to_save_detection):
    image_path = os.path.join(path_to_save_detection, images)
    detection_coordinates = get_detections_coordinates(image_path, path_to_model_wts)
    img = images.split('.')[0]
    tile_id = str(zoom_level) + '_' + img.split('_')[-1] + '_' + img.split('_')[0]
    tile_obj = Tile.tile_from_tile_id(tile_id)
    coordinates = convert_list(detection_coordinates, tile_obj.latitude_north, tile_obj.latitude_south, tile_obj.longitude_east, tile_obj.longitude_west, polygon_bounds)
    response[img] = coordinates

  list_of_coordinates = []
  for key, values in response.items():
    list_of_coordinates.extend(values)
 
  print('list_of_coordinates ', list_of_coordinates)
  for detections_result in list_of_coordinates:
    result = Result(pipeline_id=pipeline_id, gps_bounding_box=detections_result[:-1],extra_info=detections_result[-1])
    try:
      document_id = result_collection.insert_one(dict(result)).inserted_id
      print(f"Document with id {document_id} has been created")
    except:
      print("Timed out..!")

  pipeline_collection.update_one({"pipeline_id": pipeline_id}, {'$set': {'status': 'processed'}})
  
  pipeline_status = get_dict_redis(pipeline_id)
  pipeline_status[pipeline_id] = {"status":"processed"}
  set_dict_redis(pipeline_id,pipeline_status)

 