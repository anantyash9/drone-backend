
# load necessary modules
import math
import os
import urllib
import urllib.request
import numpy as np
import cv2
from shapely.geometry import Polygon
from deepforest import main
from deepforest import get_data
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def deg2num(lat_deg, lon_deg, zoom):
	lat_rad = math.radians(lat_deg)
	n = 2.0 ** zoom
	xtile = int((lon_deg + 180.0) / 360.0 * n)
	ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
	return (xtile, ytile)
	
def num2deg(xtile, ytile, zoom):
	n = 2.0 ** zoom
	lon_deg = xtile / n * 360.0 - 180.0
	lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
	lat_deg = math.degrees(lat_rad)
	return (lat_deg, lon_deg)
	

#get the range of tiles that intersect with the bounding box of the polygon	
def getTileRange(polygon, zoom):
	bnds=polygon.bounds
	xm=bnds[0]
	xmx=bnds[2]
	ym=bnds[1]
	ymx=bnds[3]
	bottomRight=(xmx,ym)
	starting=deg2num(ymx,xm, zoom)
	ending=deg2num(ym,xmx, zoom) # this will be the tiles containing the ending
	x_range=(starting[0],ending[0])
	y_range=(starting[1],ending[1])
	return(x_range,y_range)

#to get the tile as a polygon object
def getTileASpolygon(z,y,x):
	nw=num2deg(x,y,z)
	se=num2deg(x+1, y+1, z)
	xm=nw[1]
	xmx=se[1]
	ym=se[0]
	ymx=nw[0]
	tile_bound=Polygon([(xm,ym),(xmx,ym),(xmx,ymx),(xm,ymx)])
	return tile_bound	
	
#to tell if the tile intersects with the given polygon	
def doesTileIntersects(z, y, x, polygon):
	if(z<10):	#Zoom tolerance; Below these zoom levels, only check if tile intersects with bounding box of polygon
		return True
	else:
		#get the four corners
		tile=getTileASpolygon(x,y,z)
		return polygon.intersects(tile)
		
#convert the URL to get URL of Tile		
def getURL(x,y,z,url):
	u=url.replace("{x}", str(x))
	u=u.replace("{y}", str(y))
	u=u.replace("{z}", str(z))
	return u


  # polygon_bounds = convert_polygon_to_tiles(polygon_area, server_url, images_loc, zoom_level, tiles_folder)

def convert_polygon_to_tiles(polygon_area, server_url, images_loc, zoom_level, tiles_folder):
  polygon_list =[]

  for i in polygon_area:
    (x,y) = i
    polygon_list.append((x,y))	

  polygon_area=Polygon(polygon_list)

  print(polygon_area.bounds)

  tile_list=[]

  for z in range(zoom_level, zoom_level + 1):
    ranges=getTileRange(polygon_area, z)
    x_range=ranges[0]
    y_range=ranges[1]
    
    for y in range(y_range[0], y_range[1]+1):
      for x in range(x_range[0], x_range[1]+1):
        if(doesTileIntersects(x,y,z,polygon_area)):
          tile_list.append((z, y, x))
  
  tile_count=len(tile_list)


  print('Total number of Tiles: ' + str(tile_count))
  
  count=0

  for t in tile_list:
    # makeSure that folder exist; if not make it
    folderPath=os.path.join(images_loc, tiles_folder)
    filePathJ=os.path.join(folderPath,str(t[2]) + '_' + str(t[1])+'.png')
    if (not os.path.exists(folderPath)):
      os.makedirs(folderPath)
    # x,y,z,url
    url=getURL(t[2], t[1],t[0], server_url)


    urllib.request.urlretrieve(url,filePathJ)
    count=count+1
  print('finished '+str(count)+'/'+str(tile_count))

  return polygon_area

def tree_detection(path_to_images, path_to_save_detection, path_to_model_wts):
  for img in os.listdir(path_to_images):    
    file_name = os.path.join(path_to_images, img)
    path_to_save_detection = os.path.join(os.getcwd(), path_to_save_detection)
    if not os.path.exists(path_to_save_detection):
      os.mkdir(path_to_save_detection)
    save_path = os.path.join(path_to_save_detection, img)

    deepforest_model = main.deepforest.load_from_checkpoint(path_to_model_wts)
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred_after_reload = deepforest_model.predict_image(image=img, return_plot = True)

    cv2.imwrite(save_path, pred_after_reload)
    print('image saved to path: ', save_path)



def convert(OldMin,OldMax,NewMin,NewMax,OldValue):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue



def get_detections_coordinates(image_path, path_to_model_wts):
  image_path = os.path.join(os.getcwd(), image_path)
  image_path = get_data(image_path)
  print(image_path)
  im = cv2.imread(image_path)
  deepforest_model = main.deepforest.load_from_checkpoint(path_to_model_wts)
  boxes = deepforest_model.predict_image(path=image_path, return_plot = False)
  df = boxes.iloc[:,0:6]

  # Create an empty list
  row_list =[]
    
  # Iterate over each row
  for _, rows in df.iterrows():
      # Create list for the current row
      if np.all((im[int(rows.ymin): int(rows.ymax), int(rows.xmin):int(rows.xmax)] == 0)):
        flag = True
      else:
        my_list =[rows.xmin, rows.ymin, rows.xmax, rows.ymax, rows.score]
        # append the list to the final list
        row_list.append(my_list)
    
  # Print the list
  return row_list


def convert_list(detection_coordinates, latitude_north, latitude_south, longitude_east, longitude_west, polygon):
  coordinates = []
  for val in detection_coordinates:
    xmin, ymin, xmax, ymax, conf = val
  
    topleft_x= convert(0,255,longitude_west,longitude_east,xmin)
    bottomright_x= convert(0,255,longitude_west,longitude_east,xmax)
    topleft_y= convert(0,255,latitude_north,latitude_south,ymin)
    bottomright_y= convert(0,255,latitude_north,latitude_south,ymax)

    d = {}
    d['conf'] = conf

    mid_lat = (topleft_y + bottomright_y)/2
    mid_long = (topleft_x + bottomright_x)/2

    point = Point(mid_long, mid_lat) # create point

    if polygon.contains(point) and point.within(polygon):
      coordinates.append([[topleft_x,topleft_y],[bottomright_x,topleft_y],[bottomright_x,bottomright_y],[topleft_x,bottomright_y], d])

  return coordinates
