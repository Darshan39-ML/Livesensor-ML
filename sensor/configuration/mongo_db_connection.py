from pymongo import MongoClient
import os

try:
    import streamlit as st  # streamlit is optional; only used when running the Streamlit app
except Exception:
    st = None


def get_mongodb_connection():
    """Return a pymongo.MongoClient or raise an exception.

    The function will try the following (in order):
    - Streamlit secrets (if streamlit is available and secrets contain MONGODB_URI)
    - Environment variable MONGODB_URI

    It does not perform Streamlit-specific UI operations when Streamlit isn't available.
    """
    mongodb_uri = None

    # Prefer environment variable first (API / server usage)
    if os.getenv("MONGODB_URI"):
        mongodb_uri = os.getenv("MONGODB_URI")
        if st is not None and hasattr(st, "sidebar"):
            try:
                st.sidebar.info("Using Local Environment MONGODB_URI")
            except Exception:
                pass

    # Fallback to Streamlit secrets (only if env var is missing)
    if mongodb_uri is None and st is not None:
        try:
            secrets = st.secrets
        except Exception:
            secrets = None

        try:
            if secrets and isinstance(secrets, dict) and "MONGODB_URI" in secrets:
                mongodb_uri = secrets["MONGODB_URI"]
                if hasattr(st, "sidebar"):
                    try:
                        st.sidebar.success("Using Streamlit Secrets")
                    except Exception:
                        pass
        except Exception:
            # If secrets access fails, ignore and continue
            mongodb_uri = None

    if not mongodb_uri:
        # No connection string found: return None so callers can continue without DB
        return None

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    # Test the connection
    client.admin.command("ping")
    return client


class InMemoryCollection:
    def __init__(self):
        self._data = []

    def insert_many(self, records):
        self._data.extend(records)
        return {'inserted_count': len(records)}

    def find(self):
        for r in self._data:
            yield r


class InMemoryDatabase:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = InMemoryCollection()
        return self._collections[name]


class InMemoryClient:
    def __init__(self):
        self._databases = {}
        self.admin = self.Admin()

    class Admin:
        @staticmethod
        def command(x):
            # simulate ping response
            return {'ok': 1}

    def __getitem__(self, name):
        if name not in self._databases:
            self._databases[name] = InMemoryDatabase()
        return self._databases[name]



class MongoDBClient:
    """Light wrapper around pymongo.MongoClient to provide expected attributes used in the project.

    Usage:
        client = MongoDBClient(database_name=DATABASE_NAME)
        client.database  # pymongo database instance
    """

    def __init__(self, database_name: str = None, mongodb_uri: str = None):
        # Allow passing a URI directly or rely on env/secrets
        self.is_in_memory = False
        if mongodb_uri:
            # explicit uri provided: create client and test
            self.client = MongoClient(mongodb_uri)
            # test connection
            self.client.admin.command("ping")
        else:
            conn = get_mongodb_connection()
            if conn is None:
                # no uri provided and no secrets: use in-memory fallback
                self.client = InMemoryClient()
                self.is_in_memory = True
            else:
                self.client = conn

        if database_name and self.client is not None:
            self.database = self.client[database_name]
        else:
            self.database = None

    def __getitem__(self, item):
        if self.client is None:
            raise RuntimeError("MongoDB client is not configured")
        return self.client[item]

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

