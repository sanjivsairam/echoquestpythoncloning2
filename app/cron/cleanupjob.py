import os
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import shutil

# Define directories to clean
DIRECTORIES_TO_CLEAN = [
    Path("cloned"),
    Path("temp_uploads"),
    Path("processed"),
    Path("output"),
    Path("uploads"),
    Path("dubbed"),
    Path("enhanced"),
    Path("separated")
]

def cleanup_old_files():
    print("Inside cleanup old files")
    for folder in DIRECTORIES_TO_CLEAN:
        if not folder.exists():
            continue
        #print("Inside folder "+folder.name)
        items = sorted(folder.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        for old_item in items[25:]:  # Keep only the latest 10
            try:
                if old_item.is_file():
                    old_item.unlink()
                    logging.info(f"Deleted file: {old_item}")
                elif old_item.is_dir():
                    shutil.rmtree(old_item)
                    logging.info(f"Deleted folder: {old_item}")
            except Exception as e:
                logging.warning(f"Failed to delete {old_item}: {e}")

def start_cron_job():
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_old_files, 'interval', minutes=10)
    scheduler.start()
