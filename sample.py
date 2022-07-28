import collections
from telnetlib import STATUS
# from fastapi import FastAPI
from pymongo import MongoClient
import uuid

connectionString = "mongodb+srv://paperless:paperless@cluster0.58ofa.mongodb.net/vegetation?retryWrites=true&w=majority"

def insertDocument():
    detectionInfo = {
       'pipeline_id': str(uuid.uuid1()),
        'detection_type':"vegetation",
        'area': [[77.7960932114545, 12.846016963895956], [77.79617130055146, 12.846016963895956], [77.79617130055146, 12.845948705349036], [77.7960932114545, 12.845948705349036]],
        'status': "unprocessed"
    }
    document_id = collection.insert_one(detectionInfo).inserted_id
    print(f"Document with id {document_id} has been created")


def updateDocuments(pipeline_id):
    collection.update_one({"pipeline_id": pipeline_id}, {'$set': {'status': 'processed'}})
    # collection.update_many({}, {'$inc': {'section': 100}})

def readDocuments():
    # Inserting a Document
    # Reading a Collection
    # Using find function 
    # myStudents = collection.find({})
    # print(myStudents)
    # # print(myStudents)
    # for student in myStudents:
    #     print(student)
    # Using findOne function 
    # myStudent = collection.find_one({})
    # print(myStudent) 
    ids = collection.find({'task_id':'50e659e0-d908-4e93-8c7a-6e14b654c33c', 'project_id': '1'}) 
    pipline_id_list = []
    for id in ids:
        # pipline_id_list.append(id['pipeline_id'])
        print(id)
        print('===================================================================================================================================================')



if __name__ == '__main__':
    client = MongoClient(connectionString)
    # Creating a Database for a School
    db = client['vegetation']
    
    # Creating a Collection
    collection = db.vegetation
    # db.custom_models.delete_many({})
    # db.vegetation.delete_many({})
    # db.results.delete_many({})
    # for i in range(5):
    #     insertDocument()

    readDocuments()

