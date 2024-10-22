import os
import logging
import logging.config
from colorlog import ColoredFormatter


def setup_logging(default_path='app.log', default_level=logging.INFO):
    """Setup logging configuration with colorlog"""
    log_directory = os.path.dirname(default_path)
    if not os.path.exists(log_directory) and log_directory != '':
        os.makedirs(log_directory)
    colored_formatter = ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s%(reset)s',
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging.getLevelName(default_level),
                'formatter': 'colored',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': logging.getLevelName(default_level),
                'formatter': 'colored',
                'filename': default_path,
                'mode': 'w',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': logging.getLevelName(default_level),
            }
        },
        'formatters': {
            'colored': {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bg_red',
                }
            }
        }
    }

    logging.config.dictConfig(logging_config)

