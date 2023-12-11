from mindcraft.features.mood import Mood
from mindcraft.infra.engine.llm_types import LLMType
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
                 llm_type: LLMType,
                 fast: bool = False):
        """
        Instantiate a session of our game, registering a World and accepting the creation of NPCs.
        :param world_name: the name of the world
        :param embeddings: one of the supported EmbeddingsTypes, which will be used to store the lore and interactions
        :param store_type: one of the supported StoresTypes vector store
        :param llm_type: one of the supported LLMType, which will manage the generation of NPC answers
        :param fast: use vLLM fast inference (requires vLLM running in docker)
        """
        self.world = World(world_name=world_name,
                           embeddings=embeddings,
                           store_type=store_type,
                           llm_type=llm_type,
                           fast=fast)
        self.npc = dict()

    def book_to_world(self,
                      book_path: str,
                      text_splitter: TextSplitterTypes,
                      max_units: int,
                      overlap: int,
                      known_by: list[str] = None):
        """ Method that allows you to process txt files (e.g, books) and extract all the knowledge from it.
        You can set the names of the characters which you want to store this knowledge for. Otherwise, it will be
        known by all NPCs you instantiate.
        :param book_path: the path where your txt is stored
        :param text_splitter: one of the TextSplitterTypes. We recommend splitting by sentences (SentenceTextSplitter)
        :param max_units: max number of units to store in each entry in the vector store. `max_units` is an abstract
        unity measure which depends on your TextSplitterType. For example, if you use a SentenceTextSplitter,
        `max_units` will be equal to max number of sentences. If you use a TokenSplitter, then it will be tokens.
        :param overlap: how many units before and after you want to add to your chunks when splitting. This is a very
        common approach when splitting that not ony includes the splits itself, but in order to provide more context,
        it gets some text from before and after. The amount of text included from before and after is regulated by this
        parameter.
        :param known_by: a list of the name of the characters which will know about this lore. You acn leave it to None
        if this is a chronicle or something anyone knows or can easily know from the world.
        """
        self.world.book_to_world(book_path, text_splitter, max_units, overlap, known_by)

    def add_npc(self,
                character_id: str,
                description: str,
                personalities: list[Personality],
                motivations: list[Motivation],
                store_type: StoresTypes,
                mood: Mood = None,
                ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM) -> NPC:
        """
        Instantiate an NPC in the world. After the moment you instantiate the NPC, it will be available for you to
        interact with them and will remember (store) all conversations.
        :param character_id: Name of the character
        :param description: Who is the character?
        :param personalities: List of Personality objects describing PERMANENT personality features. If you look for
        non-permanent statuses or moods, use mood instead. Example of personalities: "wise", "bold", "clumsy"...
        :param motivations: List of Motivations of the character. Example: "destructing all the living"
        :param store_type: One of the StoresTypes vector store types to store your NPC lore and interactions
        :param mood: current Mood of the character. Example: "angry". If None, it will be set to a default mood.
        :param ltm_embeddings: One of EmbeddingsTypes to use to calculate embeddings from text and store in the
        VectorStore.
        :return: the NPC
        """
        npc = NPC(character_id,
                  description,
                  personalities,
                  motivations,
                  mood,
                  store_type,
                  ltm_embeddings)

        self.npc[character_id] = npc

        return npc
