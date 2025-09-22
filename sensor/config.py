# ...existing code...
from dataclasses import dataclass
import os
import pymongo

@dataclass
class EnvironmentVariable:
    mongo_db_url: str = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URL1")

env_var = EnvironmentVariable()

if not env_var.mongo_db_url:
    raise EnvironmentError(
        "MongoDB connection URL not set. Set the environment variable MONGODB_URL or MONGODB_URL1."
    )

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
# ...existing code...