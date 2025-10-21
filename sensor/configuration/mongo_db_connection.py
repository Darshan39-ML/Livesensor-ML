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
        # Do not try to show Streamlit UI here if st is None
        raise Exception("No MongoDB connection string found. Set MONGODB_URI in env or Streamlit secrets.")

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    # Test the connection
    client.admin.command("ping")
    return client


class MongoDBClient:
    """Light wrapper around pymongo.MongoClient to provide expected attributes used in the project.

    Usage:
        client = MongoDBClient(database_name=DATABASE_NAME)
        client.database  # pymongo database instance
    """

    def __init__(self, database_name: str = None, mongodb_uri: str = None):
        # Allow passing a URI directly or rely on env/secrets
        if mongodb_uri:
            self.client = MongoClient(mongodb_uri)
        else:
            self.client = get_mongodb_connection()

        if database_name:
            self.database = self.client[database_name]
        else:
            self.database = None

    def __getitem__(self, item):
        return self.client[item]

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

