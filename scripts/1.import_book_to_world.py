import argparse
import sys
import os

from infra.engine.llm_types import LLMType
from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.lore.world import World


def process_book(book_name: str, path: str):
    """
    Reads a txt book or piece of lore from `path` and stores as a World with the name of `book_name`
    :param book_name: the name of the book and world
    :param path: the path to the txt file
    """
    world = World(world_name=book_name,
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B)

    world.book_to_world(book_path=path,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processes a whole book, splits the text and adds the parts of the book as chronicles to the World')
    parser.add_argument("book_id")
    parser.add_argument("filepath")
    args = parser.parse_args()
    process_book(args.book_id, args.filepath)
