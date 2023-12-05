import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.lore.world import World
from mindcraft.settings import LOGGER_FORMAT, DATE_FORMAT

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def query_book(world_name: str,
               topic: str,
               num_results: int,
               known_by: str = None):
    """ Support script to get the pieces of lore from a stored book/world in the Vector Store
    :param topic: the topic you are interested in
    :param world_name: the name of the book/world
    :param num_results: max. num of results you want to retrieve
    :param known_by: filter by lore known only to specific NPCs. Leave it to none to retrieve common knowledge
    :return:
    """
    world = World(world_name=world_name)
    results = world.get_lore(topic, num_results, known_by)
    for d in results.documents:
        logger.info(d)
        logger.info("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retrieves from a processed book with `1.import_book_to_world.py`, a chronicle by a topic')
    parser.add_argument("world_name", help='The name of the book / world you processed with process_book')
    parser.add_argument("topic", help='The topic of the information you want to retrieve')
    parser.add_argument('-n', '--num', type=int, help='Number of results to retrieve')
    parser.add_argument('-k', '--known', type=str, help='Name of the NPC, to retrieve a chronicle only if they know it')
    args = parser.parse_args()
    query_book(args.world_name,
               args.topic,
               args.num,
               args.known)
