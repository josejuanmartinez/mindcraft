from mindcraft.settings import SEPARATOR


class Feedback:
    def __init__(self, interaction: str, answer: str):
        """
            Populates a dataset to be used in Supervised Fine-tuning as Preference Data and create your own
            NPC based on finetuned LLMs
        :param interaction:
        :param answer:
        """
        self.interaction = interaction
        self.answer = answer

    def store(self, filepath: str):
        """

        :param filepath:
        :return:
        """
        with open(filepath, "w") as f:
            f.write(SEPARATOR.join([self.interaction, self.answer]))
