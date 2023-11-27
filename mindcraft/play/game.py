from mindcraft.chronicles.world import World
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.mind.npc import NPC


class Game:
    def __init__(self, world_name: str):
        """

        :param world_name:
        """
        self.world = World(world_name=world_name)
        self.npc = dict()

    def add_npc(self, character_id: str,
                description:str,
                personalities: list[Personality],
                motivations: list[Motivation]) -> NPC:
        """

        :param character_id:
        :param description:
        :param personalities:
        :param motivations:
        :return:
        """
        npc = NPC(character_id, description, personalities, motivations)
        self.npc[character_id] = npc
        return npc


