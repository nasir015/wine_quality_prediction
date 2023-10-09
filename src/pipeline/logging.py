import logging
import os
from datetime import datetime
'''

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok= True)


LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)


logging.basicConfig(
    filename= LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)



if __name__ == "__main__":
    logging.info("Starting")
'''

import logging as lg

def logger(file_name , massage):
    lg.basicConfig(filename= file_name,
               level=lg.INFO,format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",filemode='a+')
    console_log = lg.StreamHandler()
    console_log.setLevel(lg.INFO)
    format_1 =  lg.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    console_log.setFormatter(format_1)
    lg.getLogger('').addHandler(console_log)
    logger1 = lg.getLogger("Nasir")
    logger1.info(massage)

