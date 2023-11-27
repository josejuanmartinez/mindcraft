import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes

import argparse
from mindcraft.chronicles.world import World


def process_book(book_name: str, path: str):
    """

    :param book_name:
    :param path:
    :return:
    """
    world = World(world_name=book_name, embeddings=EmbeddingsTypes.MINILM)
    world.book_to_chronicles(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processes a whole book, splits the text and adds the parts of the book as chronicles to the World')
    parser.add_argument("book_id")
    parser.add_argument("filepath")
    args = parser.parse_args()
    process_book(args.book_id, args.filepath)

