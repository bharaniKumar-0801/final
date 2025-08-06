# logger_config.py
import logging

def setup_logger(name, log_level=None):
    logger = logging.getLogger(name)
    
    # Check if logger already has handlers to avoid duplicate logs
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set default level
    if log_level == None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level)
    
    return logger
