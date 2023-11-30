from infra.engine.llm_types import LLMType
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.lore.world import World
from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.mind.npc import NPC


class Game:
    def __init__(self,
                 world_name: str,
                 embeddings: EmbeddingsTypes,
                 store_type: StoresTypes,
                 llm_type: LLMType):
        """

        :param world_name:
        """
        self.world = World(world_name=world_name,
                           embeddings=embeddings,
                           store_type=store_type,
                           llm_type=llm_type)
        self.npc = dict()

    def book_to_world(self,
                      book_path: str,
                      text_splitter: TextSplitterTypes,
                      max_units: int,
                      overlap: int,
                      known_by: list[str] = None):

        self.world.book_to_world(book_path, text_splitter, max_units, overlap, known_by)

    def add_npc(self,
                character_id: str,
                description: str,
                personalities: list[Personality],
                motivations: list[Motivation],
                store_type: StoresTypes,
                stm_capacity: int = 15,
                ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        :param character_id:
        :param description:
        :param personalities:
        :param motivations:
        :param store_type:
        :param stm_capacity:
        :param ltm_embeddings:
        :return:
        """
        npc = NPC(character_id, description, personalities, motivations, store_type, stm_capacity, ltm_embeddings)
        self.npc[character_id] = npc
        return npc


