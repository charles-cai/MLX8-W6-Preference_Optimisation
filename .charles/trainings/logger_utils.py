import logging
import colorlog

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success

def format_number(num):
    if num >= 10_000_000:
        return f"{num / 1_000_000:.0f}m"  # Format as millions without decimals
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}m"  # Format as millions with one decimal
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"  # Format as thousands with one decimal
    else:
        return str(num)
    
def setup_logger(name=__name__):
    logger = colorlog.getLogger(name)
    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s | %(message)s",
            log_colors={
                'DEBUG': 'white',
                'INFO': 'cyan',
                'SUCCESS': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger