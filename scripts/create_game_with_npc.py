import sys
import os
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.play.game import Game
from mindcraft.settings import LOGGER_FORMAT


logging.basicConfig(format=LOGGER_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def create_game_with_npc(world_name: str,
                         character_name: str,
                         character_description: str,
                         personalities: list[str],
                         motivations: list[str],
                         interaction: str):
    """
    :param character_name:
    :param character_description:
    :param world_name:
    :param personalities:
    :param motivations:
    :param interaction:
    :return:
    """
    game = Game(world_name)
    personalities = [Personality(x) for x in personalities]
    motivations = [Motivation(x) for x in motivations]
    npc = game.add_npc(character_name, character_description, personalities, motivations)
    answer, feedback = npc.react_to(interaction)
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates a game and one npc in it, with a personality and a motivation')
    parser.add_argument("world_id")
    parser.add_argument("character_name")
    parser.add_argument("character_description")
    parser.add_argument("interaction")
    parser.add_argument('-p', '--personality', action='append', nargs='*',
                        help='Specify a name of a personality feature. Example: wise')
    parser.add_argument('-m', '--motivation', action='append', nargs='*', help='Specify a name of a motivation or goal.'
                                                                               ' Example: Defender of their realm')
    args = parser.parse_args()
    create_game_with_npc(args.world_id,
                         args.character_name,
                         args.character_description,
                         [item for sublist in args.personality for item in sublist],
                         [item for sublist in args.motivation for item in sublist],
                         args.interaction)

