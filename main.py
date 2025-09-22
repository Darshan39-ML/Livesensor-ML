from sensor.exception import SensorException
import os 
import sys
import argparse
import json
from datetime import datetime
from sensor.logger import logging
from sensor.utils import dump_csv_file_to_mongodb_collection
from sensor.entity.config_entity import TrainingPipelineConfig

# def test_exception():
#     try:
#         logging.info("ki yaha p bhaiaa ek error ayegi diveision by zero wali error ")
#         a=1/0
#     except Exception as e:
#        raise SensorException(e,sys) 



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSV -> MongoDB importer (idempotent).")
    parser.add_argument("--file", "-f", default="archive/aps_failure_training_set.csv", help="Path to CSV file")
    parser.add_argument("--database", "-d", default="ineuron", help="MongoDB database name")
    parser.add_argument("--collection", "-c", default="sensor", help="MongoDB collection name")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Parse CSV but don't write to DB")
    parser.set_defaults(dry_run=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # resolve file path relative to repo root if needed
    file_path = os.path.abspath(args.file)
    database_name = args.database
    collection_name = args.collection

    try:
        summary = dump_csv_file_to_mongodb_collection(file_path, database_name, collection_name, dry_run=args.dry_run)

        logging.info("Import summary: %s", summary)

        # write summary to pipeline artifact dir so other components can find it
        pipeline_conf = TrainingPipelineConfig()
        artifacts_dir = pipeline_conf.artifact_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        out_path = os.path.join(artifacts_dir, f"import_summary_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "file": file_path, "timestamp": datetime.utcnow().isoformat()}, fh, indent=2)

        print(f"Wrote import summary to pipeline artifact dir: {out_path}")
    except Exception as e:
        raise SensorException(e, sys)



if __name__ == "__main__":
    main()




  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)