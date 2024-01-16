from typing import List


class TextSplitter:
    """Splitting text to tokens using model tokenizer."""

    def __init__(self, max_units: int, overlap: int) -> None:
        """
        :param max_units: Max number of units to be contained in a chunk
        :param overlap: Overlap window between previous and next chunks
        """
        self.max_units = max_units
        self.overlap = overlap
        pass

    def split_text(self, text: str) -> List[str]:
        """
        Applies the splitting
        :param text: Input text
        :return: a list of chunks
        """
        raise NotImplementedError()
