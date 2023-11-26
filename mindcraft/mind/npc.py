from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.infra.prompts.answer import Answer
from mindcraft.chronicles.world import World
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.memory.ltm import LTM
from mindcraft.memory.stm import STM
from mindcraft.settings import LOGGER_FORMAT

import logging

logging.basicConfig(format=LOGGER_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class NPC:
    def __init__(self,
                 character_id: str,
                 personalities: list[Personality],
                 motivations: list[Motivation],
                 stm_capacity: int = 15,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        A class managing the Non-player Character, including short-term, long-term memory, backgrounds, motivations
        to create the answer.
        :param character_id: the unique id of the character
        :param stm_capacity: the short-term memory capacity. STM stores in memory for fast retrieval this number 
        of interactions
        """
        self.character_id = character_id
        self.ltm = LTM(character_id, ltm_embeddings)
        self.stm = STM(self.ltm, stm_capacity)
        self.personalities = personalities
        self.motivations = motivations
        if not World.is_created():
            logger.warning("World has not been instantiated at this point. Make sure it's created before you call to "
                           "react")
        else:
            logger.info(f"{character_id} is now living in {World.get_instance().world_name}")

    def react_to(self, interaction: str) -> str:
        recent_events = self.stm.interactions
        long_term_events = self.ltm.remember_about(interaction)['documents'][0]
        # world_knowledge = World.get_chronicles(interaction, known_by=self.character_id)['documents']
        world_knowledge = World.get_chronicles(interaction, known_by='all')['documents'][0]
        personality_features = [x.feature for x in self.personalities]
        motivations_and_goals = [x.feature for x in self.motivations]

        print(f"Recent events: {recent_events}")
        print(f"Long-term events: {long_term_events}")
        print(f"World Knowledge: {world_knowledge}")
        print(f"Personalities: {personality_features}")
        print(f"Motivations: {motivations_and_goals}")
        prompt = Answer.create(recent_events,
                               long_term_events,
                               world_knowledge,
                               self.character_id,
                               World.get_instance().world_name,
                               interaction,
                               personality_features,
                               motivations_and_goals)

        return World.get_instance().llm(prompt)


