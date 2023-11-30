from collections import deque
from typing import List

from mindcraft.infra.splitters.text_splitter import TextSplitter


class SentenceTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""
    MAX_SPACY_CHARS = 1000000
    MAX_CHUNK_SIZE = round(MAX_SPACY_CHARS / 10)

    def __init__(self, max_units: int, overlap: int) -> None:
        """

        :param max_units:
        :param overlap:
        """
        try:
            from spacy.lang.en import English
        except ImportError:
            raise ImportError("To use Sentence Splitting, you need to install `spacy`:\n `pip install spacy`")

        super().__init__(max_units, overlap)

        nlp = English()
        nlp.add_pipe('sentencizer')
        self._nlp = nlp

    def split_text(self, text: str) -> List[str]:
        """

        :param text:
        :return:
        """
        i = 0
        finish = False
        last_chunk = deque(maxlen=self.overlap)
        while True:
            start = i*self.MAX_CHUNK_SIZE
            end = min(len(text), (i+1)*self.MAX_CHUNK_SIZE)
            # print(f"{start}-{end}")
            if end >= len(text):
                # print(f"Finished as {end}>={len(text)}")
                finish = True
            text_chunk = text[start:end]
            chunk = []
            for s in self._nlp(text_chunk).sents:
                t = s.text.strip()
                chunk.append(t)
                if len(chunk) >= self.max_units:
                    for lc in last_chunk:
                        yield "...."+lc
                    yield "\n".join(chunk)
                    for c in chunk:
                        last_chunk.append(c)
                    chunk = []

            if finish:
                if len(chunk) > 0:
                    for lc in last_chunk:
                        yield "..." + lc
                    yield "\n".join(chunk)
                    for c in chunk:
                        last_chunk.append(c)
                break
            else:
                i += 1
