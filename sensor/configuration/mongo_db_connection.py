import streamlit as st
from pymongo import MongoClient
import os

def get_mongodb_connection():
    # Try Streamlit secrets first (for deployment)
    if "MONGODB_URI" in st.secrets:
        mongodb_uri = st.secrets["MONGODB_URI"]
        st.sidebar.success("Using Streamlit Secrets")
    
    # Fallback to environment variables (for local development)
    elif os.getenv('MONGODB_URI'):
        mongodb_uri = os.getenv('MONGODB_URI')
        st.sidebar.info("🛠️ Using Local Environment")
    
    else:
        st.error(" No MongoDB connection string found")
        return None
    
    # Connect to MongoDB
    try:
        client = MongoClient(mongodb_uri)
        # Test the connection
        client.admin.command('ping')
        st.sidebar.success(" Connected to MongoDB")
        return client
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        return None

# Usage
client = get_mongodb_connection()
if client:
    db = client.my_database
    # Use your database as normal

