import os
import tempfile
import unittest

from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.lore.world import World

import pytest


@pytest.fixture
def temp_file():
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp()

    # Close the file descriptor, as we don't need it
    os.close(temp_fd)

    # Provide the temporary file path to the test
    yield temp_path

    # Clean up the temporary file after the test
    os.remove(temp_path)


def test_create_world(tmp_path):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  path=tmp_path,
                  recreate=True)

    assert world is not None

    assert not World.get_instance().fast
    assert not World.get_instance().remote

    del world


def test_create_world_fast(tmp_path):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  path=tmp_path,
                  fast=True,
                  recreate=True)

    assert world is not None

    assert World.get_instance().fast
    assert not World.get_instance().remote

    del world


def test_create_world_fast_remote(tmp_path):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  path=tmp_path,
                  recreate=True,
                  fast=True,
                  remote=True)

    assert world is not None

    assert World.get_instance().fast
    assert World.get_instance().remote

    del world


def test_import_book_to_world(tmp_path):

    temp_file = os.path.join(tmp_path, 'book.txt')

    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  path=tmp_path,
                  recreate=True)

    with open(temp_file, 'w') as file:
        file.write("In the age of Sigmur, everyone in the world is a zombie!")

    world.book_to_world(book_path=temp_file,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)

    assert world is not None


def test_import_book_to_world_and_get_lore(temp_file):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  recreate=True)

    with open(temp_file, 'w') as file:
        file.write("In the age of Sigmur, everyone in the world is a zombie!")

    world.book_to_world(book_path=temp_file,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)

    search_result = world.get_lore("Are there zombies in the world of the Age of Sigmur?", min_similarity=0.5)

    assert len(search_result.documents) > 0


if __name__ == '__main__':
    unittest.main()
