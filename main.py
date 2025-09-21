from sensor.exception import SensorException
import os
import sys
import time
import argparse
from sensor.logger import logging
from sensor.utils import dump_csv_file_to_mongodb_collection


def run_batch_import(file_path: str, database_name: str, collection_name: str, batch_size: int = 5000, dry_run: bool = True):
    """Read CSV in chunks and insert into MongoDB using the utils function per batch."""
    start = time.time()
    inserted_total = 0
    duplicates_total = 0
    errors_total = 0
    chunks = 0

    for i, chunk in enumerate(__import__('pandas').read_csv(file_path, chunksize=batch_size)):
        chunks += 1
        # write the chunk to a temp csv in-memory via to_json -> from_json to get list of dicts
        tmp_file = f"/tmp/_sensor_chunk_{i}.json"
        # use the function directly by passing the dataframe converted to csv-like file path is not supported,
        # so we'll convert chunk to records and call insert via dump_csv_file_to_mongodb_collection by
        # writing a small temp csv
        chunk.to_csv(tmp_file, index=False)
        logging.info("Processing chunk %d (rows=%d)", i + 1, len(chunk))
        res = dump_csv_file_to_mongodb_collection(tmp_file, database_name, collection_name, dry_run=dry_run)
        inserted_total += res.get('inserted', 0)
        duplicates_total += res.get('duplicates', 0)
        errors_total += res.get('errors', 0)
        logging.info("Chunk %d result: %s", i + 1, res)

    duration = time.time() - start
    logging.info("Completed import: chunks=%d, inserted=%d, duplicates=%d, errors=%d, time=%.2fs",
                 chunks, inserted_total, duplicates_total, errors_total, duration)
    return {"chunks": chunks, "inserted": inserted_total, "duplicates": duplicates_total, "errors": errors_total}

# def test_exception():
#     try:
#         logging.info("ki yaha p bhaiaa ek error ayegi diveision by zero wali error ")
#         a=1/0
#     except Exception as e:
#        raise SensorException(e,sys) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch import CSV to MongoDB')
    parser.add_argument('--file', '-f', default='/home/darshan39/Downloads/new/archive/aps_failure_training_set.csv')
    parser.add_argument('--db', '-d', default='liveclass')
    parser.add_argument('--collection', '-c', default='mongoclass')
    parser.add_argument('--batch', type=int, default=5000)
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false', help='Disable dry run and perform writes')
    args = parser.parse_args()

    logging.info('Starting batch import (dry_run=%s) ...', args.dry_run)
    try:
        result = run_batch_import(args.file, args.db, args.collection, batch_size=args.batch, dry_run=args.dry_run)
        print('Result:', result)
    except Exception as e:
        logging.exception('Import failed: %s', e)




  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)