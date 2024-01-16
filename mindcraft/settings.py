import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, '../data')
WORLD_DATA_PATH = os.path.join(DATA_PATH, 'world')
LTM_DATA_PATH = os.path.join(DATA_PATH, 'ltm')
STYLES_DATA_PATH = os.path.join(DATA_PATH, 'styles')

FAST_INFERENCE_URL = f"http://{os.environ['MINDCRAFT_HOST'] if 'MINDCRAFT_HOST' in os.environ else 'localhost'}:" \
                     f"{os.environ['MINDCRAFT_PORT'] if 'MINDCRAFT_PORT' in os.environ else '8000'}/generate"

SEPARATOR = "||"
ALL = 'all'

LOGGER_FORMAT = '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
DATE_FORMAT = '%d-%m-%Y:%H:%M:%S'
