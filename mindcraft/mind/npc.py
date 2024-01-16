from mindcraft.memory.summarizer_types import SummarizerTypes
from mindcraft.memory.stm import STM
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.sft.feedback import Feedback
from mindcraft.features.motivation import Motivation
from mindcraft.features.personality import Personality
from mindcraft.lore.world import World
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.memory.ltm import LTM
from mindcraft.settings import LOGGER_FORMAT, DATE_FORMAT
from mindcraft.features.mood import Mood
from mindcraft.styles.conversational_style import ConversationalStyle

import logging


logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class NPC:
    def __init__(self,
                 character_name: str,
                 description: str,
                 personalities: list[Personality],
                 motivations: list[Motivation],
                 mood: Mood,
                 store_type: StoresTypes,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM,
                 stm_capacity: int = 5,
                 stm_summarizer: SummarizerTypes = SummarizerTypes.T5_SMALL,
                 stm_max_summary_length: int = 230,
                 stm_min_summary_length: int = 30):
        """
        A class managing the Non-player Character, including short-term, long-term memory, backgrounds, motivations
        to create the answer.
        :param character_name: the unique id of the character
        :param description: a short description of who your character in the world is
        :param personalities: a list of personalities that permanently define the character
        (if it's a current state then use it in `moods`)
        :param motivations: a list of motivations the character has
        :param mood: current mood of the character. They can change over the time.
        :param store_type: VectorStore from StoresTypes you prefer to use.
        :param ltm_embeddings: embeddings from EmbeddingsTypes you prefer to use
        :param stm_capacity: How many interactions from ltm to store
        :param stm_summarizer: One of `SummarizerTypes` to use for including the summary of last interactions
        :param stm_max_summary_length: max length of the summary
        :param stm_min_summary_length: min length of the summary
        """
        self._character_name = character_name
        self._description = description
        self._ltm = LTM(store_type, character_name, ltm_embeddings)
        self._stm = STM(self._ltm, stm_capacity, stm_summarizer, stm_max_summary_length, stm_min_summary_length)
        self._personalities = personalities
        self._motivations = motivations
        self._mood = mood
        self._conversational_style = ConversationalStyle(store_type, character_name, ltm_embeddings)
        self._last_interaction = ""
        self._last_answer = ""

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
        """
        res_dict = dict()
        exact_match = f"{keyword} {self._character_name}"
        results = World.get_lore(exact_match, 25, self._character_name, exact_match)
        for d in results.documents:
            mood = input(f"Found this interaction of {self._character_name}:\n{'='*10}\n{d}\n{'='*10}\n"
                         f"Enter a mood name OR `d` for default mood OR `i` to ignore OR `q` to quit:")
            if mood == 'q':
                break
            elif mood == 'i':
                continue
            else:
                if mood == 'd':
                    mood = Mood.DEFAULT
                mood = Mood(mood)
                question = input(f"What could have been told to {self._character_name} to answer that?:")
                if mood not in res_dict:
                    res_dict[mood.feature] = list()
                res_dict[mood.feature].append(d)
                # Save as Conversational Style
                self._conversational_style.memorize(d, mood)
                # Save to feedback file
                Feedback(self._character_name, mood, self._conversational_style, question, d).accept()
        return res_dict

    def extract_conversational_styles_talking_to_user(self):
        """
        Creates conversations where the npc talks in the World to the user.
        """
        while True:
            interaction = input(f"Say something to {self._character_name} or `q` to quit:")
            if interaction == 'q':
                break
            else:

                answer, feedback = self.react_to(interaction,
                                                 min_similarity=0.85,
                                                 ltm_num_results=3,
                                                 world_num_results=7,
                                                 max_tokens=200)
                print(answer)
                mood = input(
                    f"Enter a mood for this answer, `d` for default, `i` to ignore this answer and `q` to quit:")
                if mood == 'q':
                    break
                elif mood == 'i':
                    continue
                else:
                    if mood == 'd':
                        mood = Mood.DEFAULT

                    feedback.accept(mood=Mood(mood))

    def react_to(self,
                 interaction: str,
                 min_similarity: float = 0.85,
                 ltm_num_results: int = 3,
                 world_num_results: int = 10,
                 max_tokens: int = 250,
                 temperature: float = 0.8) -> tuple[str, Feedback]:
        """
        Produces a reaction/answer to something you say to an NPC. It will use the lore of the world + the short memory
        + the long-term-memory + the personality, motivations and moods to generate a response.

        Use an iterator to process as it will yield chunks as they come in from the LLM.


        :param interaction: the interaction / question you tell/ask the NPC
        :param min_similarity: minimum similarity score to filter out irrelevant information
        :param ltm_num_results: max number of results to retrieve from Long-term memory
        :param world_num_results: max number of results to retrieve from World Lore
        :param max_tokens: max_tokens of the answer
        :param temperature: temperature or how creative the answer should be
        :return: a tuple with the text of the answer and a Feedback object, in case you want to use to review the answer
        and provide feedback to the model, for training future npc-based LLMs.
        """
        self._last_interaction = interaction

        memories = self._ltm.remember_about(interaction,
                                            num_results=ltm_num_results,
                                            min_similarity=min_similarity).documents

        world_knowledge = World.get_lore(interaction,
                                         known_by=self._character_name,
                                         num_results=world_num_results,
                                         min_similarity=min_similarity).documents

        personalities = [x.feature for x in self._personalities] if self._personalities is not None else []
        motivations = [x.feature for x in self._motivations] if self._motivations is not None else []
        mood = self._mood.feature if self._mood is not None else Mood.DEFAULT

        conversational_style = self._conversational_style.retrieve_interaction_by_mood(mood).documents

        # I create the prompt
        prompt = World.create_prompt(memories,
                                     world_knowledge,
                                     self._character_name,
                                     interaction,
                                     personalities,
                                     motivations,
                                     conversational_style,
                                     mood)

        logger.info(prompt)

        chunks = []
        for chunk in World.retrieve_answer_from_llm(prompt,
                                                    max_tokens=max_tokens,
                                                    do_sample=True,
                                                    temperature=temperature):
            yield chunk
            chunks.append(chunk)

        self._last_answer = "".join(chunks)
        self._ltm.memorize(self._last_answer, self._mood)

    def retrieve_feedback_to_finetune(self) -> Feedback:
        """
        Retrieves a Feedback object given the last interaction, which can help you store the data for revision and
        eventually train of your own finetuned LLM
        :return: a Feedback object
        """
        if self._last_answer is None \
                or len(self._last_answer) < 1 \
                or self._last_interaction is None \
                or len(self._last_interaction) < 1:
            raise Exception("Feedback unavailable (no interaction or answer found as last)")

        return Feedback(self._character_name, self._mood, self._conversational_style, self._last_interaction,
                        self._last_answer)

    def change_mood(self, mood: str):
        """
        Changes the mood of the character.
        :param mood: string defining the new mood
        """
        self._mood = Mood(mood)

    @property
    def character_name(self):
        """ Getter of the `character_name` property"""
        return self._character_name

    @property
    def mood(self):
        """ Getter of the `mood` property"""
        return self._mood

    @property
    def conversational_style(self):
        """ Getter of the `conversational_style` property"""
        return self._conversational_style

    def add_npc_to_world(self):
        """
        Instantiate an NPC in the world. After the moment you instantiate the NPC, it will be available for you to
        interact with them and will remember (store) all conversations.
        """
        if World.get_instance() is None:
            raise Exception("World not found. Please instantiate a ´World´ first.")

        World.get_instance().npcs[self.character_name] = self
        print(f"`{self.character_name}` now lives in `{World.get_instance().world_name}`")
