from typing import List


class TextSplitter:
    """Splitting text to tokens using model tokenizer."""

    def __init__(self, max_units: int, overlap: int) -> None:
        """

        :param max_units:
        :param overlap:
        """
        self.max_units = max_units
        self.overlap = overlap
        pass

    def split_text(self, text: str) -> List[str]:
        """

        :param text:
        :return:
        """
        raise NotImplementedError()
