import pandas as pd
import logging
import json
from typing import Dict, Mapping, Any as _Any
from pymongo.errors import BulkWriteError, PyMongoError
from sensor.config import mongo_client


def _stable_id_for_record(record: Mapping[_Any, _Any]) -> str:
    """Create a stable id for a record by hashing its JSON representation.

    This makes repeated imports idempotent when the same CSV data is imported again.
    """
    import hashlib

    payload = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dump_csv_file_to_mongodb_collection(
    file_path: str,
    database_name: str,
    collection_name: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Read CSV and insert records into MongoDB collection.

    Returns a summary dict with counts: to_write, inserted, duplicates, errors.
    """
    summary = {"to_write": 0, "inserted": 0, "duplicates": 0, "errors": 0}

    try:
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        # convert to list of dicts in a stable order
        records = df.to_dict(orient="records")

        # assign stable _id when missing so re-inserts are idempotent
        for rec in records:
            if "_id" not in rec:
                # rec is a mapping-like dict; cast to Dict[str, Any] for hashing
                rec["_id"] = _stable_id_for_record(dict(rec))

        summary["to_write"] = len(records)

        if dry_run:
            logging.info("Dry run enabled: %d records parsed, no DB write.", len(records))
            return summary

        if not records:
            return summary

        try:
            result = mongo_client[database_name][collection_name].insert_many(records, ordered=False)
            summary["inserted"] = len(result.inserted_ids)
        except BulkWriteError as bwe:
            # count inserted vs duplicate key errors
            details = bwe.details or {}
            write_errors = details.get("writeErrors", [])
            dup_count = sum(1 for e in write_errors if e.get("code") == 11000)
            summary["duplicates"] = dup_count
            # inserted count is total - write_errors
            n = details.get("nInserted") or 0
            summary["inserted"] = n
            logging.warning("Bulk write completed with errors: %s", str(bwe))
        except PyMongoError as pe:
            logging.error("PyMongo error during insert: %s", pe)
            summary["errors"] += 1

    except FileNotFoundError:
        logging.error("CSV file not found: %s", file_path)
        summary["errors"] = 1
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty: %s", file_path)
        summary["errors"] = 1
    # Let unexpected exceptions propagate so callers can handle them.
    return summary
