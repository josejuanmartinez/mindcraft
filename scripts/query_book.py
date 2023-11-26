import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.chronicles.world import World
from mindcraft.settings import LOGGER_FORMAT


logging.basicConfig(format=LOGGER_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def query_book(book_name: str,
               topic: str,
               num_results: int,
               known_by: str):
    """
    :param topic:
    :param book_name:
    :param num_results:
    :param known_by:
    :return:
    """
    world = World(world_name=book_name)
    results = world.get_chronicles(topic, num_results, known_by)
    for document in results['documents']:
        for d in document:
            logger.info(d)
            logger.info("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retrieves from a processed book with `process_book.py`, a chronicle by a topic')
    parser.add_argument("book_name", help='The name of the book / world you processed with process_book')
    parser.add_argument("topic", help='The topic of the information you want to retrieve')
    parser.add_argument('-n', '--num', type=int, help='Number of results to retrieve')
    parser.add_argument('-k', '--known', type=str, help='Name of the NPC, to retrieve a chronicle only if they know it')
    args = parser.parse_args()
    query_book(args.book_name,
               args.topic,
               args.num,
               args.known)
