from typing import List, Optional
import tiktoken


class TextSplitter:
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int,
        tokens_per_chunk: int,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
    ) -> None:
        """

        :param chunk_overlap:
        :param tokens_per_chunk:
        :param encoding_name:
        :param model_name:
        """
        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._chunk_overlap = chunk_overlap
        self._tokens_per_chunk = tokens_per_chunk

    def split_text(self, text: str) -> List[str]:
        """

        :param text:
        :return:
        """
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(_text)

        input_ids = _encode(text)
        start_idx = 0
        cur_idx = min(start_idx + self._tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            yield self._tokenizer.decode(chunk_ids)
            start_idx += self._tokens_per_chunk - self._chunk_overlap
            cur_idx = min(start_idx + self._tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
