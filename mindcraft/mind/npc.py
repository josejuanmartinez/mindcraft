from mindcraft.infra.sft.feedback import Feedback
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
                 character_name: str,
                 description: str,
                 personalities: list[Personality],
                 motivations: list[Motivation],
                 stm_capacity: int = 15,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        A class managing the Non-player Character, including short-term, long-term memory, backgrounds, motivations
        to create the answer.
        :param character_name: the unique id of the character
        :param description: a short description of who your character in the world is
        :param personalities:
        :param motivations:
        :param stm_capacity: the short-term memory capacity. STM stores in memory for fast retrieval this number
        of interactions
        :param ltm_embeddings:
        """
        self.character_name = character_name
        self.description = description
        self.ltm = LTM(character_name, ltm_embeddings)
        self.stm = STM(self.ltm, stm_capacity)
        self.personalities = personalities
        self.motivations = motivations
        if not World.is_created():
            logger.warning("World has not been instantiated at this point. Make sure it's created before you call to "
                           "react")
        else:
            logger.info(f"{character_name} is now living in {World.get_instance().world_name}")

    def react_to(self, interaction: str) -> tuple[str, Feedback]:
        """

        :param interaction:
        :return:
        """
        recent_events = self.stm.interactions
        long_term_events = self.ltm.remember_about(interaction)['documents'][0]
        # world_knowledge = World.get_chronicles(interaction, known_by=self.character_id)['documents']
        world_knowledge = World.get_chronicles(interaction, known_by='all')['documents'][0]
        personality_features = [x.feature for x in self.personalities]
        motivations_and_goals = [x.feature for x in self.motivations]

        prompt = Answer.create(recent_events,
                               long_term_events,
                               world_knowledge,
                               self.character_name,
                               World.get_instance().world_name,
                               interaction,
                               personality_features,
                               motivations_and_goals)

        answer = World.get_instance().llm(prompt)
        return answer, Feedback(interaction, answer)
