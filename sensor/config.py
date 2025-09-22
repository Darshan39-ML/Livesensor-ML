# ...existing code...
from dataclasses import dataclass
import os
import pymongo

try:
    # dotenv is optional in some environments; catch ImportError only
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv isn't installed or available; continue using environment variables
    pass


@dataclass
class EnvironmentVariable:
    mongo_db_url: str = (
        os.getenv('MONGODB_URI')
        or os.getenv('MONGODB_URL')
        or os.getenv('MONGODB_URI1')
        or os.getenv('MONGODB_URL1')
        or os.getenv('MONGO_URL')
        or ''
    )


env_var = EnvironmentVariable()

if not env_var.mongo_db_url:
    raise EnvironmentError(
        "MongoDB connection URL not set. Set one of: MONGODB_URI, MONGODB_URL, MONGODB_URI1, MONGODB_URL1, or MONGO_URL."
    )

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
# ...existing code...