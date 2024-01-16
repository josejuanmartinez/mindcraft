import os
import tempfile
import unittest

from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.lore.world import World

from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.features.mood import Mood

from mindcraft.mind.npc import NPC

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


@pytest.fixture
def world_fixture(temp_file):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  recreate=True,
                  fast=False,
                  remote=False,
                  streaming=False)

    with open(temp_file, 'w') as file:
        file.write("In the age of Sigmur, everyone in the world is a zombie!")

    world.book_to_world(book_path=temp_file,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)

    return world


@pytest.fixture
def fast_world_fixture(temp_file):
    world = World(world_name="TheAgeOfSigmur",
                  embeddings=EmbeddingsTypes.MINILM,
                  store_type=StoresTypes.CHROMA,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  recreate=True,
                  fast=True,
                  remote=False,
                  streaming=False)

    with open(temp_file, 'w') as file:
        file.write("In the age of Sigmur, everyone in the world is a zombie!")

    world.book_to_world(book_path=temp_file,
                        text_splitter=TextSplitterTypes.SENTENCE_SPLITTER,
                        max_units=3,
                        overlap=1)

    return world


def test_npc_is_created(world_fixture):
    name = "Zombie Leader"
    description = "The Zombie Leader leaving in the Age of Sigmur World"
    personalities = [Personality(x) for x in ['evil']]
    motivations = [Motivation(x) for x in ['Eating human flesh']]
    mood = Mood('angry')

    zombie = NPC(name,
                 description,
                 personalities,
                 motivations,
                 mood,
                 StoresTypes.CHROMA,
                 EmbeddingsTypes.MINILM)

    assert zombie is not None


def test_npc_is_added_to_world(world_fixture):
    name = "Zombie Leader"
    description = "The Zombie Leader leaving in the Age of Sigmur World"
    personalities = [Personality(x) for x in ['evil']]
    motivations = [Motivation(x) for x in ['Eating human flesh']]
    mood = Mood('angry')

    zombie = NPC(name,
                 description,
                 personalities,
                 motivations,
                 mood,
                 StoresTypes.CHROMA,
                 EmbeddingsTypes.MINILM)

    assert zombie is not None
    zombie.add_npc_to_world()
    assert zombie.character_name in World.get_instance().npcs


def test_npc_conversation_local(world_fixture):

    name = "Zombie Leader"
    description = "The Zombie Leader leaving in the Age of Sigmur World"
    personalities = [Personality(x) for x in ['evil']]
    motivations = [Motivation(x) for x in ['Eating human flesh']]
    mood = Mood('angry')

    zombie = NPC(name,
                 description,
                 personalities,
                 motivations,
                 mood,
                 StoresTypes.CHROMA,
                 EmbeddingsTypes.MINILM)

    assert zombie is not None
    zombie.add_npc_to_world()
    assert zombie.character_name in World.get_instance().npcs
    answer_iter = zombie.react_to("What is your main motivation?",
                                  min_similarity=0.85,
                                  ltm_num_results=3,
                                  world_num_results=7,
                                  max_tokens=600)
    for answer in answer_iter:
        print(answer)
        assert answer is not None


if __name__ == '__main__':
    unittest.main()
