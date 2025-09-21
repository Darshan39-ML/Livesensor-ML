import pandas as pd
import numpy as np
import logging
import json
import hashlib
from typing import List, Dict, Any
from pymongo.errors import BulkWriteError, DuplicateKeyError
from sensor.config import mongo_client


def _stable_id_for_record(record: Dict[str, Any]) -> str:
    """Create a stable hash id for a record based on its JSON representation."""
    # Use sorted keys to ensure deterministic output
    payload = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def dump_csv_file_to_mongodb_collection(
    file_path: str,
    database_name: str,
    collection_name: str,
    dry_run: bool = False,
    id_field: str = "_id",
) -> Dict[str, int]:
    """Load a CSV and insert into MongoDB collection.

    This function is idempotent: if documents don't have an `_id`, it assigns a stable
    SHA1-based `_id` so re-running the import won't create duplicates. Use `dry_run=True`
    to validate without performing network writes.

    Returns a summary dict with counts: inserted, duplicates, errors.
    """
    try:
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        # Convert dataframe to list of dicts
        records: List[Dict[str, Any]] = json.loads(df.T.to_json())
        # json.loads(df.T.to_json()) returns a dict-of-dicts with numeric keys; extract values
        if isinstance(records, dict):
            records = list(records.values())

        # Ensure each record has a stable _id
        for rec in records:
            if id_field not in rec or rec[id_field] in (None, ""):
                rec[id_field] = _stable_id_for_record(rec)

        if dry_run:
            logging.info("Dry run enabled — no writes will be performed. Records to write: %d", len(records))
            return {"to_write": len(records), "inserted": 0, "duplicates": 0, "errors": 0}

        coll = mongo_client[database_name][collection_name]
        # Perform unordered bulk insert so duplicates don't abort the whole batch
        try:
            result = coll.insert_many(records, ordered=False)
            inserted = len(result.inserted_ids)
            return {"to_write": len(records), "inserted": inserted, "duplicates": 0, "errors": 0}
        except BulkWriteError as bwe:
            # count inserted and duplicate key errors
            write_result = bwe.details
            inserted = write_result.get("nInserted", 0)
            # duplicate key errors are in writeErrors with code 11000
            dup_count = 0
            for err in write_result.get("writeErrors", []):
                if err.get("code") == 11000:
                    dup_count += 1
            logging.warning("BulkWriteError: inserted=%d, duplicates=%d", inserted, dup_count)
            return {"to_write": len(records), "inserted": inserted, "duplicates": dup_count, "errors": 1}

    except Exception as e:
        logging.exception("Failed to dump csv to mongodb: %s", e)
        return {"to_write": 0, "inserted": 0, "duplicates": 0, "errors": 1}
