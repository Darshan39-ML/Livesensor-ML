from dotenv import load_dotenv
import pymongo
from sensor.constant.database import DATABASE_NAME
import certifi
ca = certifi.where()
from sensor.constant.env_variable import MONGODB_URL_KEY
import os 
import logging 

# Fix: provide the path to your .env file
load_dotenv(dotenv_path='/home/darshan39/Downloads/Sensor-ML/.env') # e.g. .env is in a folder called 'config' so it would be 'config/.env'
client = None
class MongoDBClient:


    def __init__(self, database_name=DATABASE_NAME) -> None:
        
        try:
            
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                logging.info("Retrieved MongoDB URL: %s", mongo_db_url)
# OR using the slightly newer style:
# logging.info("Retrieved MongoDB URL: {}", mongo_db_url)

                if not mongo_db_url:
                    raise ValueError(f"MongoDB URL not found. Please set {MONGODB_URL_KEY} in your environment variables or .env file.")

                if "mongodb+srv://darshan39:Krishn_4@cluster0.puwtldn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)   #TLS/SSl 
                
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            logging.error('Error initializing MongoDB client: %s!', e)
            raise
