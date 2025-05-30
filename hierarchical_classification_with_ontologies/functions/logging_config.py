import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

def create_log_file(name):
    logName = f"{os.path.join(os.getenv("LOG_FOLDER"))}{datetime.now().strftime("%d%m%Y_%I%M%S%p")}_{str(name)}.log"
    logging.basicConfig(filename=logName, filemode='w', format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO)