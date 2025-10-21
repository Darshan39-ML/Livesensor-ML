import pytest
import asyncio

import main


def test_health_endpoint(monkeypatch):
    # monkeypatch ModelResolver.is_model_exists to True
    class DummyResolver:
        def __init__(self, model_dir=None):
            pass

        def is_model_exists(self):
            return True

    monkeypatch.setattr(main, 'ModelResolver', DummyResolver)

    # monkeypatch MongoDBClient to a dummy that returns ok
    class DummyMongo:
        def __init__(self, database_name=None, mongodb_uri=None):
            pass

        def close(self):
            pass

        @property
        def client(self):
            class Admin:
                def command(self, x):
                    return True

            class C:
                def __init__(self):
                    self.admin = Admin()

            return C()

    monkeypatch.setattr(main, 'MongoDBClient', DummyMongo)

    result = asyncio.run(main.health())
    assert result['model'] == 'available'
    assert result['mongodb'] == 'ok'
