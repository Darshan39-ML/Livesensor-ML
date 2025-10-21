import pytest
import os

from sensor.configuration.mongo_db_connection import get_mongodb_connection, MongoDBClient


def test_get_mongodb_connection_env(monkeypatch):
    monkeypatch.setenv('MONGODB_URI', 'mongodb://invalid:27017')
    # should raise on ping since invalid uri, but ensure function attempts to connect
    with pytest.raises(Exception):
        get_mongodb_connection()


def test_mongo_wrapper_no_env(monkeypatch):
    monkeypatch.delenv('MONGODB_URI', raising=False)
    # create a dummy streamlit secrets and monkeypatch st object
    class DummyST:
        secrets = {'MONGODB_URI': 'mongodb://invalid:27017'}

        class sidebar:
            @staticmethod
            def info(x):
                pass

            @staticmethod
            def success(x):
                pass

    import sys
    monkeypatch.setitem(sys.modules, 'streamlit', DummyST)
    # Reload the module under test to pick up monkeypatched streamlit
    import importlib
    import sensor.configuration.mongo_db_connection as m
    importlib.reload(m)

    with pytest.raises(Exception):
        m.MongoDBClient()
