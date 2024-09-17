import colorlog
import logging

# Configure logger
def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


# Create a logger instance
logger = logging.getLogger(__name__)
addLoggingLevel("TRACE", logging.DEBUG - 5)
logger.setLevel(logging.TRACE)  


handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter("%(log_color)s%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | >>> %(message)s",
                                      datefmt="%Y-%m-%dT%H:%M:%SZ",
                                      log_colors={
                                          'TRACE': 'purple',
                                          'DEBUG': 'cyan',
                                          'INFO': 'green',
                                          'WARNING': 'yellow',
                                          'ERROR': 'red',
                                          'CRITICAL': 'red,bg_white',
                                      })
handler.setFormatter(formatter)
logger.addHandler(handler)

# Exporting the logger
def get_logger():
    return logger