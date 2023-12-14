import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.mind.npc import NPC
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.settings import LOGGER_FORMAT, DATE_FORMAT
from mindcraft.features.mood import Mood
from mindcraft.lore.world import World

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def create_game_with_npc(world_name: str,
                         character_name: str,
                         character_description: str,
                         personalities: list[str],
                         motivations: list[str],
                         interaction: str,
                         mood: str = None):
    """ Example script to showcase how to instantiate a game with an NPC in it
    :param character_name: name of the character
    :param character_description: description of the character
    :param world_name: name of the world / book
    :param personalities: list of strings describing the NPC personality (permanent features, e.g: "wise")
    :param motivations: list of strings describing the motivations of the NPC (e.g: "destroying the evil")
    :param interaction: question / topic you want the NPC to react on
    :param mood: string describing the current mood of the NPC (e.g: "angry". leave to None to fallback to default)
    :return:
    """
    # World should have been instantiated with lore. See 1.import_book_to_world.py
    world = World(world_name=world_name,
                  store_type=StoresTypes.CHROMA,
                  embeddings=EmbeddingsTypes.MINILM,
                  llm_type=LLMType.ZEPHYR7B_AWQ,
                  fast=False)

    personalities = [Personality(x) for x in personalities]
    motivations = [Motivation(x) for x in motivations]
    mood = Mood(mood)

    npc = NPC(character_name,
              character_description,
              personalities,
              motivations,
              mood,
              StoresTypes.CHROMA,
              EmbeddingsTypes.MINILM)

    npc.add_npc_to_world()

    answer, _ = npc.react_to(interaction,
                             min_similarity=0.85,
                             ltm_num_results=3,
                             world_num_results=7,
                             max_tokens=200,
                             temperature=0.4)

    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates a game and an NPC in it that answers to one question')
    parser.add_argument("world_name")
    parser.add_argument("character_name")
    parser.add_argument("character_description")
    parser.add_argument("interaction")
    parser.add_argument('-p', '--personality', default=[], action='append', nargs='*',
                        help='Specify a name of a personality feature. Example: wise')
    parser.add_argument('-m', '--motivation', default=[], action='append', nargs='*',
                        help='Specify a name of a motivation or goal. Example: Defender of their realm')
    parser.add_argument('-o', '--mood', nargs='?', help='Optional - Specify a current mood. Example: angry')
    args = parser.parse_args()
    create_game_with_npc(args.world_name,
                         args.character_name,
                         args.character_description,
                         [item for sublist in args.personality for item in sublist],
                         [item for sublist in args.motivation for item in sublist],
                         args.interaction,
                         args.mood)
