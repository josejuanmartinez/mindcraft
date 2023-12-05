import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, '../data')
WORLD_DATA_PATH = os.path.join(DATA_PATH, 'world')
LTM_DATA_PATH = os.path.join(DATA_PATH, 'ltm')
STYLES_DATA_PATH = os.path.join(DATA_PATH, 'styles')

SEPARATOR = "||"
ALL = 'all'

LOGGER_FORMAT = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
DATE_FORMAT = '%d-%m-%Y:%H:%M:%S'
