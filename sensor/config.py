from dataclasses import dataclass
import os
import pymongo
from typing import Optional

try:
    # optional dependency: python-dotenv to load a local .env during development
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or failed to load; proceed with environment variables
    pass


@dataclass
class EnvironmentVariable:
    # Support several common names to be resilient to existing setups
    mongo_db_url: Optional[str] = (os.getenv('MONGODB_URI') or
                                   os.getenv('MONGODB_URI1') or
                                   os.getenv('MONGODB_URL') or
                                   os.getenv('MONGODB_URL1') or
                                   os.getenv('MONGO_URL') or
                                   None)


env_var = EnvironmentVariable()

if not env_var.mongo_db_url:
    raise EnvironmentError(
        "MONGODB connection URI not found. Set one of: MONGODB_URI, MONGODB_URI1, MONGODB_URL, MONGODB_URL1, or MONGO_URL in your environment or .env file."
    )

# Use the provided URI to create the client. This will attempt to connect lazily.
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)