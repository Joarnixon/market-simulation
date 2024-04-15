import logging
import os
import shutil


class Logger:
    def __init__(self, log_folder='logs'):
        self.log_folder = log_folder
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.makedirs(self.log_folder, exist_ok=True)
        self.loggers = {}

    def get_logger(self, instance_name) -> logging.Logger:
        if instance_name not in self.loggers:
            logger = logging.getLogger(instance_name)
            logger.setLevel(logging.INFO)
            os.makedirs(self.log_folder, exist_ok=True)
            log_file = os.path.join(self.log_folder, f"{instance_name}_log.txt")
            logger.addHandler(logging.FileHandler(log_file))
            self.loggers[instance_name] = logger
        return self.loggers[instance_name]

    def log_event(self, instance_name, message):
        logger = self.get_logger(instance_name)
        logger.info(message)