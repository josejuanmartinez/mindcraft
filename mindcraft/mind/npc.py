from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.sft.feedback import Feedback
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.infra.prompts.answer import Answer
from mindcraft.lore.world import World
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.memory.ltm import LTM
from mindcraft.memory.stm import STM
from mindcraft.settings import LOGGER_FORMAT, ALL
from mindcraft.features.mood import Mood

import logging

from styles.conversational_style import ConversationalStyle

logging.basicConfig(format=LOGGER_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class NPC:
    def __init__(self,
                 character_name: str,
                 description: str,
                 personalities: list[Personality],
                 motivations: list[Motivation],
                 store_type: StoresTypes,
                 stm_capacity: int = 15,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        A class managing the Non-player Character, including short-term, long-term memory, backgrounds, motivations
        to create the answer.
        :param character_name: the unique id of the character
        :param description: a short description of who your character in the world is
        :param personalities:
        :param motivations:
        :param store_type:
        :param stm_capacity: the short-term memory capacity. STM stores in memory for fast retrieval this number
        of interactions
        :param ltm_embeddings:
        """
        self.character_name = character_name
        self.description = description
        self.ltm = LTM(store_type, character_name, ltm_embeddings)
        self.stm = STM(self.ltm, stm_capacity)
        self.personalities = personalities
        self.motivations = motivations
        self.conversational_style = ConversationalStyle(store_type, character_name, ltm_embeddings)
        if not World.is_created():
            logger.warning("World has not been instantiated at this point. Make sure it's created before you call to "
                           "react")
        else:
            logger.info(f"{character_name} is now living in {World.get_instance().world_name}")
            logger.info(f"If you want to extract conversations from the character and assign them to mood using "
                        f"the created world, run `extract_conversational_styles_from_world`")

    def extract_conversational_styles_from_world(self, keyword: str = 'said') -> {}:
        """
        Finds conversations where the npc talks in the World and asks for a mood to store as an example of
        conversational style. The pattern will be constructed by using a keyword and the name of the character.
        For example: `said Galadriel`.
        NOTE: This is not a semantic search since it would retrieve many non-conversational agents. Make sure to chose
        properly your keyword.
        :param keyword: string to look for in exact match. Default: `said`
        :return:
        """
        res_dict = dict()
        exact_match = f"{keyword} {self.character_name}"
        results = World.get_lore(exact_match, 25, self.character_name, exact_match)
        for document in results['documents']:
            for d in document:
                mood = input(f"Found this interaction of {self.character_name}:\n{'='*10}\n{d}\n{'='*10}\n"
                             f"Enter a mood name OR `d` for default mood OR `i` to ignore OR `q` to quit:")
                if mood == 'q':
                    break
                elif mood == 'i':
                    continue
                else:
                    if mood not in res_dict:
                        res_dict[mood] = list()
                    res_dict[mood].append(d)
                    self.conversational_style.memorize(d, Mood(mood))
        return res_dict

    def react_to(self, interaction: str) -> tuple[str, Feedback]:
        """

        :param interaction:
        :return:
        """
        recent_events = self.stm.interactions
        long_term_events = self.ltm.remember_about(interaction)['documents'][0]
        # world_knowledge = World.get_chronicles(interaction, known_by=self.character_id)['documents']
        world_knowledge = World.get_lore(interaction, known_by=ALL)['documents'][0]
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
