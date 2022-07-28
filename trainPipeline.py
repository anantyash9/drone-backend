import numpy as np

import os

import cv2
import sys
import shutil

from shapely.geometry import Polygon, box
from utils import convert_polygon_to_tiles, convert, convert_list
from tiles import Tile
from PIL import Image
from config import custom_models, result_collection, pipeline_collection, odm_link, token
from models import Result
from rediscli import set_dict_redis, get_dict_redis


def yolov5_data_format(detection_type):

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset'))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type)):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data'))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images'))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images/train')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images/train'))
    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images/valid')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/images/valid'))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels'))

    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels/train')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels/train'))
    if not os.path.exists(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels/valid')):
        os.mkdir(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data/labels/valid'))

    with open(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data', 'voc.names'), 'a') as f:
        f.write(detection_type)

    yaml_txt = 'train: {}\nval: {}\n\n# number of classes\nnc: 1\n\n# class names\nnames: [{}]'.format(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data', 'train.txt'),
                                                                                                    os.path.join(os.getcwd(), 'trainingset', detection_type, 'data', 'train.txt'), 
                                                                                                    detection_type)
    with open(os.path.join(os.getcwd(), 'trainingset', detection_type, 'data', 'detection.yaml'), 'w') as f:
        f.write(yaml_txt)

def complete_image(detection_type, polygon_area, project_id, task_id, zoom_level, preprocess):

  # step 1
  server_url = odm_link +'api/projects/'+ project_id + '/tasks/' + task_id + '/orthophoto/tiles/{z}/{x}/{y}.png?jwt=' +  token 
  
  # change path as per location
  loc=os.path.join(os.getcwd(), "images") 
  folder_name = detection_type 
  
  folderPath=os.path.join(loc, folder_name)
  if (os.path.exists(folderPath)):
     shutil.rmtree(folderPath)
  # zoom_level = 19 # adjust zoom level. Take as a parameter

  stArea = convert_polygon_to_tiles(polygon_area, server_url, loc, zoom_level, folder_name)
  
  path_to_save_detection = os.path.join(loc, folder_name)
  latitude_north, latitude_south, longitude_east, longitude_west = 0, sys.maxsize*1.0, 0, sys.maxsize*1.0
  
  # for loop
  for images in os.listdir(path_to_save_detection):
    img = images.split('.')[0]
    tile_id = str(zoom_level) + '_' + img.split('_')[-1] + '_' + img.split('_')[0]
    obj = Tile.tile_from_tile_id(tile_id)
    latitude_north = max(latitude_north, obj.latitude_north)
    latitude_south = min(latitude_south, obj.latitude_south)
    longitude_east = max(longitude_east, obj.longitude_east)
    longitude_west = min(longitude_west, obj.longitude_west)

  imgs = []
  y = []
  x = []

  for images in os.listdir(path_to_save_detection):
    if images.endswith('.png'):
      imgs.append(images)
      y.append(int(images.split('_')[-1].split('.')[0]))
      x.append(int(images.split('_')[0]))
  imgs.sort()
  print(x)
  new_img_width = (max(x) - min(x) + 1)*256
  new_img_ht = (max(y) - min(y) + 1)*256

  print('new dims ', (new_img_width, new_img_ht))
  dst = Image.new('RGB', (new_img_width, new_img_ht))

  for i in range(min(x), max(x)+1):
    for j in range(min(y), max(y)+1):
      image = f'{i}_{j}.png'
      if image in imgs:
        img = Image.open(os.path.join(path_to_save_detection, image))
        pos_x = (i - min(x))*256
        pos_y = (j - min(y))*256
        dst.paste(img, (pos_x, pos_y))

  prev_img = imgs[0]
  dst.save(os.path.join(path_to_save_detection, prev_img))

  preprocess.append((max(x) - min(x) + 1))
  preprocess.append((max(y) - min(y) + 1))
  preprocess.append(stArea)

  return (os.path.join(path_to_save_detection, prev_img), latitude_north, latitude_south, longitude_east, longitude_west)


def preprocess(detection_type, polygon_area, list_of_geojson, count, project_id = '1', task_id = 'ae87bd1d-17e7-4430-bd41-574b8ce82554'): 
  
  preprocess = []

  # adjust zoom level according to the area occupied by object
  ratio  = 0.0
  zoom_level = 19
  while ratio < 0.10 and zoom_level < 25:
    preprocess = []
    path_to_complete_image, latitude_north, latitude_south, longitude_east, longitude_west = complete_image(detection_type, 
                                                                                                        polygon_area, 
                                                                                                        project_id, 
                                                                                                        task_id,
                                                                                                        zoom_level,
                                                                                                        preprocess)

    tile_poly = Polygon([
                      [longitude_west, latitude_north],
                      [longitude_east, latitude_north],
                      [longitude_east, latitude_south],
                      [longitude_west, latitude_south]
    ])

    ratio = preprocess[-1].area/tile_poly.area

    print('zoom_level ', zoom_level, 'ratin ', ratio)
    zoom_level += 1

  prev_img = path_to_complete_image.split('/')[-1]
  path_to_save_detection = '/'

  for folder in path_to_complete_image.split('/')[:-1]:
    path_to_save_detection = os.path.join(path_to_save_detection, folder)

  print('path_to_complete_image ', path_to_complete_image, prev_img, path_to_save_detection)
  
  # New co=ordinates for current object in the form of pixels
  # latitude_north, latitude_south, 0, 255, old_south
  OldMin, OldMax, NewMin, NewMax, OldValue = latitude_north, latitude_south, 0, (preprocess[1]*256 - 1), preprocess[-1].bounds[1]
  new_pixel_latitude_south = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

  # latitude_north, latitude_south
  OldMin, OldMax, NewMin, NewMax, OldValue = latitude_north, latitude_south, 0, (preprocess[1]*256 - 1), preprocess[-1].bounds[3]
  new_pixel_latitude_north = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

  # longitude_west, longitude_east
  OldMin, OldMax, NewMin, NewMax, OldValue = longitude_west, longitude_east, 0, (preprocess[0]*256 - 1), preprocess[-1].bounds[0]
  new_pixel_longitude_west = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

  # longitude_west, longitude_east
  OldMin, OldMax, NewMin, NewMax, OldValue = longitude_west, longitude_east, 0, (preprocess[0]*256 - 1), preprocess[-1].bounds[2]
  new_pixel_longitude_east = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))


  print('north(ymin), south(ymax), west(xmin), east(xmax): ',new_pixel_latitude_north, new_pixel_latitude_south, new_pixel_longitude_west, new_pixel_longitude_east)


  # creating yolov5 format
  new_coords = [[new_pixel_latitude_north, new_pixel_latitude_south, new_pixel_longitude_west, new_pixel_longitude_east]]

  # Co-ordinates of cropped image for current object in form of pixels
  img = cv2.imread(path_to_complete_image)
  padding = 500 # no. of pixels
  crop_xmin = (new_pixel_longitude_west - padding) if (new_pixel_longitude_west - padding) > 0 else 0
  crop_xmax = (new_pixel_longitude_east + padding) if (new_pixel_longitude_west + padding) < img.shape[1] else img.shape[1]
  crop_ymin = (new_pixel_latitude_north - padding) if (new_pixel_latitude_north - padding) > 0 else 0
  crop_ymax = (new_pixel_latitude_south + padding) if (new_pixel_latitude_south + padding) < img.shape[0] else img.shape[0]

  cv2.imwrite(path_to_complete_image, img[crop_ymin:crop_ymax, crop_xmin:crop_xmax])

  # modify the new coordinates of current object
  new_coords[0][0] -= crop_ymin
  new_coords[0][1] -= crop_ymin
  new_coords[0][2] -= crop_xmin
  new_coords[0][3] -= crop_xmin

  # modifying the cropped pixels
  crop_ymax -= crop_ymin
  crop_xmax -= crop_xmin

  # (OR)
  img = cv2.imread(path_to_complete_image)

  # Geo co-ordinates of cropped image for current object
  topleft_x = convert(0,img.shape[1],longitude_west,longitude_east,crop_xmin)
  bottomright_x = convert(0,img.shape[1],longitude_west,longitude_east,crop_xmax)
  topleft_y = convert(0,img.shape[0],latitude_north,latitude_south,crop_ymin)
  bottomright_y = convert(0,img.shape[0],latitude_north,latitude_south,crop_ymax)

  crop_coords = [topleft_y, bottomright_y, topleft_x, bottomright_x]

  # Box(geo coords) of the cropped complete image
  b = box(crop_coords[2], crop_coords[0], crop_coords[-1], crop_coords[1])

  #for c in range(count+1, len(list_of_geojson)):
  for c in range(0, len(list_of_geojson)):
    
    # Other polygons in the train/valid dataset 
    poly_b = Polygon(list_of_geojson[c])
    if b.contains(poly_b) and (c != int(count)):

        # New co=ordinates for object in the form of pixels
        # latitude_north, latitude_south, 0, 255, old_south
        OldMin, OldMax, NewMin, NewMax, OldValue = latitude_north, latitude_south, 0, (img.shape[0] - 1), poly_b.bounds[1]
        new_pixel_latitude_south = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

        # latitude_north, latitude_south
        OldMin, OldMax, NewMin, NewMax, OldValue = latitude_north, latitude_south, 0, (img.shape[0] - 1), poly_b.bounds[3]
        new_pixel_latitude_north = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

        # longitude_west, longitude_east
        OldMin, OldMax, NewMin, NewMax, OldValue = longitude_west, longitude_east, 0, (img.shape[1] - 1), poly_b.bounds[0]
        new_pixel_longitude_west = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

        # longitude_west, longitude_east
        OldMin, OldMax, NewMin, NewMax, OldValue = longitude_west, longitude_east, 0, (img.shape[1] - 1), poly_b.bounds[2]
        new_pixel_longitude_east = int(convert(OldMin, OldMax, NewMin, NewMax, OldValue))

        new_coords.append([new_pixel_latitude_north, new_pixel_latitude_south, new_pixel_longitude_west, new_pixel_longitude_east])

  
  return (new_coords, path_to_save_detection, prev_img, zoom_level)

def yolo_train_valid(detection_type, train_flag, path_to_save_detection, prev_img, new_coords, iter):
  path = 'train' if train_flag else 'valid'
  path = 'data/images/{}/'.format(path)
  path = os.path.join(os.getcwd(), 'trainingset', detection_type, path,'{}.png'.format(iter)) + '\n' # change 1.png to iterable name.png
  
  print('image path ', path)
  
  file = 'train' if train_flag else 'valid'
  file = '{}.txt'.format(file)
  file_path = os.path.join(os.getcwd(), 'trainingset', detection_type, 'data', file)
  with open(file_path, 'a') as f:
    f.write(path)
  
  print('file_path ', file_path)
  
  img = cv2.imread(os.path.join(path_to_save_detection, prev_img))
  print(os.path.join(path_to_save_detection, prev_img))
  
  image_size = img.shape
  cv2.imwrite(path[:-1], img)

  # labels
  path = 'train' if train_flag else 'valid'
  path = 'data/labels/{}/'.format(path)
  path = os.path.join(os.getcwd(), 'trainingset', detection_type, path,'{}.txt'.format(iter)) # change 1.png to iterable name.png
  print('label path ', path)
  
  for i in range(len(new_coords)):
    new_pixel_latitude_north, new_pixel_latitude_south, new_pixel_longitude_west, new_pixel_longitude_east = new_coords[i]
    with open(path, 'a') as f:
        class_id = 0

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (new_pixel_longitude_west + new_pixel_longitude_east) / 2 
        b_center_y = (new_pixel_latitude_north + new_pixel_latitude_south) / 2
        b_width    = (new_pixel_longitude_east- new_pixel_longitude_west)
        b_height   = (new_pixel_latitude_south- new_pixel_latitude_north)

        # Normalise the co-ordinates by the dimensions of the image
        image_h, image_w, image_c = image_size  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h

        text = '{} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(class_id, b_center_x, b_center_y, b_width, b_height)

        f.write(text)

        
def train_pipeline(training_info):
    training_info['status'] = 'processing'
    pipeline_collection.update_one({"pipeline_id": training_info['training_id']}, {'$set': {'status': 'processing'}})

    
    yolov5_data_format(training_info['detection_type'])

    if len(training_info['list_of_geojson']) == 2:
        train_valid_split = 1
    elif len(training_info['list_of_geojson']) > 2:
        train_valid_split = len(training_info['list_of_geojson'])//3


    count = 0
    avg_zoom_level = 0
    while count < len(training_info['list_of_geojson']):
        polygon_area = training_info['list_of_geojson'][count]
        if (count < (len(training_info['list_of_geojson']) - train_valid_split)):
            train_flag = True
            new_coords, path_to_save_detection, prev_img, zoom_level = preprocess(training_info['detection_type'], 
                                                                     polygon_area, 
                                                                     training_info['list_of_geojson'],
                                                                     count,
                                                                     training_info['project_id'], 
                                                                     training_info['task_id'],
                                                            )
            
            yolo_train_valid(training_info['detection_type'], 
                            train_flag, 
                            path_to_save_detection, 
                            prev_img, 
                            new_coords, 
                            str(count))
        else:
            train_flag = False
            new_coords, path_to_save_detection, prev_img, zoom_level = preprocess(training_info['detection_type'], 
                                                                     polygon_area, 
                                                                     training_info['list_of_geojson'],
                                                                     count,
                                                                     training_info['project_id'], 
                                                                     training_info['task_id']
                                                            )
            
            yolo_train_valid(training_info['detection_type'], 
                            train_flag, 
                            path_to_save_detection, 
                            prev_img, 
                            new_coords, 
                            str(count))
        shutil.rmtree(path_to_save_detection)
        avg_zoom_level += zoom_level
        count += 1
    avg_zoom_level = avg_zoom_level//len(training_info['list_of_geojson']) 
    print('Preprocessing Done...!')
   
    # training yolov5
    path_to_yolo_trainpy = os.path.join(os.getcwd(), 'yolov5/train.py')
    imgsz = '640'
    batch_zs = '2'
    eps = '15'
    data_path_to_yaml = os.path.join(os.getcwd(), 'trainingset', training_info['detection_type'], 'data', 'detection.yaml')
    wts = 'yolov5l.pt'
    name_of_folder = training_info['detection_type']
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cmd_to_train = 'python {} --img {} --batch {} --epochs {} --data {} --weights {} --name {} --cache'.format(path_to_yolo_trainpy,
                                                                                                    imgsz,
                                                                                                    batch_zs,
                                                                                                    eps,
                                                                                                    data_path_to_yaml,
                                                                                                    wts,
                                                                                                    name_of_folder)
    # train.run(imgsz=640, data=data_path_to_yaml, epochs = eps, weights = 'yolov5s.pt', batch_size = 4, device="cpu")
    response = os.system(cmd_to_train)
   

    if response == 0:
        print('yolov5 training successful..!')
        training_info['status'] = 'processed'
        insert_doc = {}
        insert_doc['training_id'] = training_info['training_id']
        insert_doc['detection_type'] = training_info['detection_type']
        insert_doc['path_to_model_weights'] = os.path.join(os.getcwd(), 'yolov5/runs/train', training_info['detection_type'], 'weights', 'best.pt')
        insert_doc['zoom_level'] = avg_zoom_level
        insert_doc['status'] = 'processed'
        document_id = custom_models.insert_one(insert_doc).inserted_id
        print(f"Custom Model Document with id {document_id} has been created")

        pipeline_collection.update_one({"pipeline_id": training_info['training_id']}, {'$set': {'status': 'processed'}})
        training_info['status'] = 'processed'


    else:
        print('response', response)
        pipeline_collection.update_one({"pipeline_id": training_info['training_id']}, {'$set': {'status': str(response)}})

def custom_model_inference(detection_info):

    detection_info['status'] = 'processsing'

    pipeline_status = get_dict_redis(detection_info['pipeline_id'])
    pipeline_status[detection_info['pipeline_id']] = {"status":"processing"}
    set_dict_redis(detection_info['pipeline_id'],pipeline_status)

    pipeline_collection.update_one({"pipeline_id": detection_info['pipeline_id']}, {'$set': {'status': 'processing'}})

    # Fetching the data of custom model from mongdb
    custom_model_info = custom_models.find_one({'detection_type': detection_info['model_name']})
    print('custom_model_info ', custom_model_info)

    preprocess = []
    zoom_level = custom_model_info['zoom_level']
    path_to_complete_image, latitude_north, latitude_south, longitude_east, longitude_west = complete_image(detection_info['model_name'], 
                                                                                                            detection_info['area'], 
                                                                                                            detection_info['project_id'], 
                                                                                                            detection_info['task_id'],
                                                                                                            zoom_level,
                                                                                                            preprocess)
    print('path_to_complete_image', path_to_complete_image)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    
    weights = os.path.join(os.getcwd(), 'yolov5', 'runs', 'train', detection_info['model_name'], 'weights', 'best.pt')
    img_sz = '640'
    conf_threshold =  ' 0.25'
    name_of_folder = detection_info['model_name']
    cmd_line = 'python /app/yolov5/detect.py --weights {} --img {} --conf {} --source {} --name {} --save-txt --save-conf'.format(weights, 
                                                                                                                                img_sz, 
                                                                                                                                conf_threshold, 
                                                                                                                                path_to_complete_image, 
                                                                                                                                name_of_folder)
    
    # Training the yolov5 model
    os.system(cmd_line)
    print('Inference completed..!')
    
    path_to_detect = os.path.join(os.getcwd(), 'yolov5', 'runs', 'detect')
    folders_in_detect = []
    for folders in os.listdir(path_to_detect):
        folders_in_detect.append(folders)
    folders_in_detect.sort()
    print('folders_in_detect', folders_in_detect)

    path_to_labels = os.path.join(path_to_detect, folders_in_detect[-1], 'labels')
    
    detection_coordinates = []
    for file in os.listdir(path_to_labels):
        print('file ', file)
        with open(os.path.join(path_to_labels, file), 'r') as f:
            txt = f.read().split('\n')
            class_label, xmin, ymin, xmax, ymax, conf = list(map(float, txt[0].split(' ')))
            detection_coordinates.append([class_label, xmin, ymin, xmax, ymax, conf])
    if len(detection_coordinates) > 0:
        coordinates = convert_list(detection_coordinates, latitude_north, latitude_south, longitude_east, longitude_west, detection_info['area'])
    
        for detections_result in coordinates:
            result = Result(pipeline_id=detection_info['pipeline_id'], gps_bounding_box=detections_result[:-1],extra_info=detections_result[-1])
            try:
                document_id = result_collection.insert_one(dict(result)).inserted_id
                print(f"Document with id {document_id} has been created")
            except:
                print("Timed out..!")
    
    detection_info['status'] = 'processed'
    pipeline_status = get_dict_redis(detection_info['pipeline_id'])
    pipeline_status[detection_info['pipeline_id']] = {"status":"processed"}
    set_dict_redis(detection_info['pipeline_id'],pipeline_status)

    pipeline_collection.update_one({"pipeline_id": detection_info['pipeline_id']}, {'$set': {'status': 'processed'}})
    






