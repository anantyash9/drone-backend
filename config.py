import requests
from pymongo import MongoClient

client = MongoClient("mongodb+srv://paperless:paperless@cluster0.58ofa.mongodb.net/vegetation?retryWrites=true&w=majority")

db = client['vegetation']

# Creating a Collection
pipeline_collection = db.vegetation
result_collection = db.results
custom_models = db.custom_models


odm_link = "http://stgrog.ddns.net:8000/"

auth_response = requests.post(odm_link + 'api/token-auth/', 
                      data={'username': 'paperless',
                            'password': 'paperless'}).json()
token = auth_response['token']
  

#change path as per location
path_to_model_wts = './checkpoint_20.pl'

