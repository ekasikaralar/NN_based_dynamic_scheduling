#!/usr/bin/env python3

import logging
import subprocess
from pathlib import Path
from os import walk

"""
Database to CSV Exporter

This script is designed to automate the process of exporting data from database files (MDB format) 
located within a specified directory structure to CSV files. It navigates through each subdirectory 
in a given database directory, processes each database file, and exports specific tables to CSV files.
"""

# Constants
DATABASE_DIR = Path("USBank/database")
OUTPUT_DIR = Path("out")
EXCLUDED_FILE = "garbage.mdb"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mdb_export(database_path, table, output_file):
    try:
        subprocess.run(["mdb-export", str(database_path), table, ">", str(output_file)], check=True)
        logging.info(f"Exported {table} from {database_path} to {output_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error exporting {database_path}: {e}")

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    for folder in next(walk(DATABASE_DIR), (None, None, []))[1]:
        logging.info("----------------------------------------------------------")
        logging.info(folder)

        path = DATABASE_DIR / folder
        for file in next(walk(path), (None, None, []))[2]:
            if file != EXCLUDED_FILE:
                logging.info(file)
                name = Path(file).stem
                database_path = path / file

                for table in ["cust_subcalls", "agent_records", "agent_events"]:
                    output_file = OUTPUT_DIR / f"{name}_{table}.csv"
                    mdb_export(database_path, table, output_file)

if __name__ == "__main__":
    main()
