from sensor.exception import SensorException
import os
import sys
import argparse
from sensor.logger import logging
from sensor.utils import dump_csv_file_to_mongodb_collection
from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig

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
        # Create pipeline config and ingestion config
        pipeline_conf = TrainingPipelineConfig()
        ingestion_conf = DataIngestionConfig(pipeline_conf)

        # Use the ingestion config's feature store path for the output file
        summary = dump_csv_file_to_mongodb_collection(
            file_path, database_name, collection_name, dry_run=args.dry_run
        )

        logging.info("Import summary: %s", summary)

        # Ensure pipeline artifact dir exists
        os.makedirs(pipeline_conf.artifact_dir, exist_ok=True)

        # Try to get header line from the input CSV to populate placeholders
        header_line = None
        try:
            with open(file_path, "r", encoding="utf-8") as fin:
                first_line = fin.readline()
                if first_line and ("," in first_line or "\t" in first_line):
                    header_line = first_line.strip()
        except (OSError, UnicodeDecodeError):
            # If the file can't be read for IO or decoding reasons, continue with no header
            header_line = None

        created_files = []
        for path in (
            ingestion_conf.feature_store_file_path,
            ingestion_conf.training_file_path,
            ingestion_conf.testing_file_path,
        ):
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            # write header or placeholder
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    if header_line:
                        fh.write(header_line + "\n")
                    else:
                        fh.write("# placeholder file created by main.py\n")
                created_files.append(path)
            except OSError as e:
                logging.error("Failed to create file %s: %s", path, e)

        print("Created files:")
        for p in created_files:
            print(" -", p)
    except Exception as e:
        # preserve original traceback
        raise SensorException(e, sys) from e



if __name__ == "__main__":
    main()




  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)