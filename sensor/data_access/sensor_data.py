import sys
from typing import Optional

import numpy as np
import pandas as pd 
import json
import sensor.configuration.mongo_db_connection as mongo_client
from sensor.exception import SensorException
from sensor.constant.database import DATABASE_NAME



class SensorData:
    def __init__(self, 
                 collection_name: str,
                 database_name: str = DATABASE_NAME
                 ) -> None:
        """
        This class shall be used to read the data from MongoDB and 
        write it to a pandas dataframe
        """
        try:
            self.database_name = database_name
            self.collection_name = collection_name
            self.mongo_client = mongo_client.MongoDBClient(database_name=self.database_name)
            self.collection = self.mongo_client.database[self.collection_name]
        except Exception as e:
            raise SensorException(e, sys) from e

    def save_csv_file(self, file_path, collection_name:str, database_name: Optional[str] = None):
        
        try:
            data_frame = pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[collection_name]
            collection.insert_many(records)
        except Exception as e:
            raise SensorException(e, sys) from e


    def get_collection_as_dataframe(self, database_name: Optional[str] = None) -> pd.DataFrame:

        try:
            if database_name is None:
                collection = self.mongo_client.database[self.collection_name]
            else:
                collection = self.mongo_client.database[self.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise SensorException(e, sys) from e