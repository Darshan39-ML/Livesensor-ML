from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient as _MongoClient, errors as _pymongo_errors
try:
    from sensor.constant.database import DATABASE_NAME
except ImportError:
    # DATABASE_NAME missing; fall back to a safe default for tests
    DATABASE_NAME = 'test'
import os
import certifi
ca = certifi.where()
from sensor.constant.env_variable import MONGODB_URL_KEY
import logging

load_dotenv()

class MongoDBClient:
    # type hint for the class-level client
    client: _MongoClient | None = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # Read the connection URL from env; support fallback keys if needed
                mongo_db_url = os.getenv(MONGODB_URL_KEY) or os.getenv('MONGODB_URI') or os.getenv('MONGODB_URL')

                if not mongo_db_url:
                    raise EnvironmentError(
                        f"MongoDB connection URL not found. Set the environment variable {MONGODB_URL_KEY} or MONGODB_URI"
                    )

                # Mask sensitive info when logging: show only scheme and host
                from urllib.parse import urlparse

                parsed = urlparse(mongo_db_url)
                host_display = parsed.hostname or 'unknown-host'

                logging.info("Initializing MongoDB client for host: %s", host_display)

                # Use short server selection timeout to fail fast on bad configs
                server_selection_timeout_ms = 5000

                # For local connections, don't pass TLS CA file
                scheme = getattr(parsed, 'scheme', None)
                if 'localhost' in host_display or scheme == 'mongodb':
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, serverSelectionTimeoutMS=server_selection_timeout_ms)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca, serverSelectionTimeoutMS=server_selection_timeout_ms)

            self.client = MongoDBClient.client
            # self.client is a MongoClient, subscriptable to access databases
            assert self.client is not None
            # static type checkers may not understand pymongo's client subscripting.
            # use typing.cast to make intent explicit.
            # Use get_database to avoid static-type complaints about subscripting
            self.database = self.client.get_database(database_name)
            self.database_name = database_name
        except (_pymongo_errors.PyMongoError, OSError) as e:
            logging.error("Error initializing MongoDBClient: %s", e)
            raise
