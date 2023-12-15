from transformers import pipeline

from mindcraft.memory.summarizer_types import SummarizerTypes
from mindcraft.memory.ltm import LTM


class STM:
    def __init__(self,
                 ltm: LTM,
                 capacity: int = 5,
                 summarizer: SummarizerTypes = SummarizerTypes.T5_SMALL,
                 max_summary_length: int = 230,
                 min_summary_length: int = 30):
        """ Short-term memory is used to include always a summarized version of what has been discussed lately
        :param ltm: The Long-Term Memory object
        :param capacity: How many interactions from ltm to store
        :param summarizer: One of `SummarizerTypes` to use for including the summary of last interactions
        :param max_summary_length: max length of the summary
        :param min_summary_length: min length of the summary
        """
        self._ltm = ltm
        self._summarizer = summarizer
        self._summarizer_model = pipeline("summarization", model=str(summarizer.value))
        self._max_summary_length = max_summary_length
        self._min_summary_length = min_summary_length
        self._capacity = capacity
        self._summary = self.initialize_summary()

    def initialize_summary(self) -> str:
        """
        Retrieves `self.capacity` last interactions from LTM and stores summarized
        :return: the summary
        """
        search_result = self._ltm.get_last_interactions(self._capacity)
        text = ".".join(search_result.documents)
        if len(text) < self._min_summary_length:
            return text
        text = self._summarizer_model(text,
                                      max_length=min(len(text), self._max_summary_length),
                                      min_length=self._min_summary_length,
                                      do_sample=False)
        return text[0]['summary_text']

    def refresh_summary(self, last_interaction: str):
        """
        Refresh the summary with the last interaction
        :param last_interaction: last answer of the NPC
        :return: summary
        """
        self.summary = ".".join([self.initialize_summary(), last_interaction])
        return self.summary

    @property
    def summary(self):
        """ retrieves the summary property"""
        return self._summary

    @summary.setter
    def summary(self, value: str):
        """ sets the summary property"""
        self._summary = value



