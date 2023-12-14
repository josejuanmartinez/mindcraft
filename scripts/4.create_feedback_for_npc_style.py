import sys
import os
import argparse
import logging

from features.mood import Mood

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.play.game import Game
from mindcraft.settings import LOGGER_FORMAT, DATE_FORMAT

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feedback_for_npc_style(world_name: str,
                                  character_name: str,
                                  character_description: str,
                                  personalities: list[str],
                                  motivations: list[str]):
    """ Example script to showcase how to create a feedback file to train your own (smaller?) LLM based on:
     -1) conversations from a book that you tag with a mood;
     -2) own generated interactions with an LLM of your choice;
    :param character_name: name of the character
    :param character_description: description of the character
    :param world_name: name of the world / book
    :param personalities: list of strings describing the NPC personality (permanent features, e.g: "wise")
    :param motivations: list of strings describing the motivations of the NPC (e.g: "destroying the evil")
    :return:
    """
    # World should have been instantiated with lore. See 1.import_book_to_world.py
    game = Game(world_name=world_name,
                store_type=StoresTypes.CHROMA,
                embeddings=EmbeddingsTypes.MINILM,
                llm_type=LLMType.ZEPHYR7B_AWQ)
    personalities = [Personality(x) for x in personalities]
    motivations = [Motivation(x) for x in motivations]

    npc = game.add_npc_to_world(character_name,
                                character_description,
                                personalities,
                                motivations,
                                StoresTypes.CHROMA,
                                ltm_embeddings=EmbeddingsTypes.MINILM)

    logger.info(f"Extracting conversations of {character_name} from the world...")
    npc.extract_conversational_styles_from_world()

    logger.info(f"Generating conversations by talking to you!")
    npc.extract_conversational_styles_talking_to_user()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates a game and an NPC in it that answers to one question')
    parser.add_argument("world_name")
    parser.add_argument("character_name")
    parser.add_argument("character_description")
    parser.add_argument('-p', '--personality', action='append', nargs='*',
                        help='Specify a name of a personality feature. Example: wise')
    parser.add_argument('-m', '--motivation', action='append', nargs='*', help='Specify a name of a motivation or goal.'
                                                                               ' Example: Defender of their realm')
    args = parser.parse_args()
    create_feedback_for_npc_style(args.world_name,
                                  args.character_name,
                                  args.character_description,
                                  [item for sublist in args.personality for item in sublist],
                                  [item for sublist in args.motivation for item in sublist])
