from typing import List

from mindcraft.infra.splitters.text_splitter import TextSplitter


class TokenTextSplitter(TextSplitter):
    def __init__(self,
                 max_units: int,
                 overlap: int,
                 encoding_name: str = "gpt2") -> None:
        """Splitting text to tokens using model tokenizer.
        :param max_units: Number of tokens to include in a chunk
        :param overlap: Number of tokens to overlap with previous and following chunks
        :param encoding_name:
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError("To use GPT2 token-based text splitting, you need to install `tiktoken`:\n"
                              "`pip install tiktoken`")
        super().__init__(max_units, overlap)
        self._encoding_name = encoding_name
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def split_text(self, text: str) -> List[str]:
        """

        :param text:
        :return:
        """
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(_text)

        input_ids = _encode(text)
        start_idx = 0
        cur_idx = min(start_idx + self.max_units, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            yield self._tokenizer.decode(chunk_ids)
            start_idx += self.max_units - self.overlap
            cur_idx = min(start_idx + self.max_units, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
