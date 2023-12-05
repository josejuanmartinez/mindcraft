import logging
import os

from mindcraft.features.mood import Mood
from mindcraft.settings import SEPARATOR, LOGGER_FORMAT, DATE_FORMAT, STYLES_DATA_PATH
from mindcraft.styles.conversational_style import ConversationalStyle

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class Feedback:
    def __init__(self,
                 character_name: str,
                 mood: Mood,
                 conversational_style: ConversationalStyle,
                 interaction: str, answer: str):
        """
            Populates a dataset to be used in Supervised Fine-tuning as Preference Data and create your own
            NPC based on finetuned LLMs
        :param character_name: name of the NPC
        :param mood: mood string (e.g., 'angry')
        :param conversational_style: Conversational Style Object of the NPC, which will be updated using this
        interaction
        :param interaction: question/topic asked to the NPC
        :param answer: answer from the NPC
        """
        self._character_name = character_name
        self._mood = mood
        self._conversational_style = conversational_style
        self._interaction = interaction
        self._answer = answer

    def accept(self,
               folder: str = STYLES_DATA_PATH,
               separator: str = SEPARATOR,
               mood: Mood = None):
        """
        Accepts this interaction as valid for training purposes. It will populate it to a CSV and also store it as a
        conversational style for the character for future interactions.
        :param folder: csv path where to save the feedback
        :param separator: csv separator. Default: SEPARATOR (||)
        :param mood: Mood to overwrite (if not set, self._npc.mood will be taken)
        """
        if mood is None:
            mood = self._mood
        with open(os.path.join(folder, self._character_name, "sft.csv"), "a") as f:
            f.write(separator.join([self._character_name if self._character_name is not None else '',
                                    mood.feature if mood is not None else Mood.DEFAULT,
                                    self._interaction.encode("unicode_escape").decode("utf-8"),
                                    self._answer.encode("unicode_escape").decode("utf-8")]))
            f.write("\n")
            logger.info(f"Interaction appended to {folder}")

        self._conversational_style.memorize(self._answer, self._mood)
