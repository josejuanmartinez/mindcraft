from typing import Type, Union, Iterator, List

from mindcraft.infra.prompts.prompt import Prompt
from mindcraft import settings
from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.vectorstore.search_results import SearchResult
from mindcraft.infra.splitters.sentence_text_splitter import SentenceTextSplitter
from mindcraft.infra.splitters.token_text_splitter import TokenTextSplitter
from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.vectorstore.store import Store
from mindcraft.infra.engine.local_llm import LocalLLM
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.engine.remote_vllm import RemoteVLLM
from mindcraft.infra.engine.local_vllm import LocalVLLM
from mindcraft.settings import SEPARATOR, LOGGER_FORMAT, WORLD_DATA_PATH, ALL, FAST_INFERENCE_URL

import logging

logging.basicConfig(format=LOGGER_FORMAT, datefmt=settings.DATE_FORMAT, level=logging.ERROR)
logger = logging.getLogger(__name__)


class World:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        World story. It stores everything that happened in a world.
        They are kept in the vector store.
        Not every NPC will know what happened in the world. Metadata will be used.
        :param world_id: the unique `id` of the character
        :param store: element of type StoresTypes
        :param ltm_embeddings: Embeddings to use in LTM in the Vector Store.
        :param llm_type: Embeddings to use in LTM in the Vector Store.
        :param world_path: Custom path where to store the data of the world. If not set, falls back to WORLD_DATA_PATH
        :param fast: use vLLM fast inference (requires vLLM running in docker)
        :param fast: use vLLM fast inference in cases vLLM is not in local but served in an external server.
         In this case, an HTTP connection will be established
        """
        if 'world_name' not in kwargs:
            raise Exception("To instantiate a world, please add the name of the world in `world_name`")

        if 'store_type' not in kwargs:
            raise Exception("`store_type` not found in World() initializer")

        if 'embeddings' not in kwargs:
            logger.warning("`embeddings` not found in World() initializer. "
                           f"Initializing to {str(EmbeddingsTypes.MINILM.value)}")

        if 'fast' in kwargs and not isinstance(kwargs.get('fast'), bool):
            raise Exception("The value for `fast` param should be True or False")

        if 'remote' in kwargs and not isinstance(kwargs.get('remote'), bool):
            raise Exception("The value for `remote` param should be True or False")

        if 'streaming' in kwargs and not isinstance(kwargs.get('streaming'), bool):
            raise Exception("The value for `streaming` param should be True or False")

        if 'llm_type' not in kwargs:
            logger.warning(f"`llm_type` not found in World() initializer. Initializing to {LLMType.ZEPHYR7B_AWQ}")
        elif not isinstance(kwargs.get('llm_type'), LLMType):
            raise Exception(f"`llm_type` should be of type  `LLMType`")

        create_world = False
        destroying_world = False

        if cls._instance is None:
            create_world = True
        elif ('recreate' in kwargs and kwargs.get('recreate')) or(kwargs.get('world_name') != cls._instance.world_name):
            create_world = True
            destroying_world = True

        if create_world:
            if destroying_world:
                logger.info(f"Changing world from {cls._instance.world_name} to {kwargs.get('world_name')}")

            cls._instance = super().__new__(cls)
            cls._instance._world_name = kwargs.get('world_name')
            cls._instance._embeddings = kwargs.get('embeddings') if 'embeddings' in kwargs else EmbeddingsTypes.MINILM
            cls._instance._store_type = kwargs.get('store_type')
            cls._instance._llm_type = kwargs.get('llm_type') if 'llm_type' in kwargs else LLMType.ZEPHYR7B_AWQ
            cls._instance._world_data_path = kwargs.get('path') if 'path' in kwargs else WORLD_DATA_PATH
            cls._instance._fast = kwargs.get('fast') if 'fast' in kwargs else False
            cls._instance._remote = kwargs.get('remote') if cls._instance._fast and 'remote' in kwargs else False
            cls._instance._streaming = kwargs.get('streaming') \
                if cls._instance._remote and 'streaming' in kwargs else False
            cls._instance._llm = None
            cls._instance._npcs = dict()

            match cls._instance._store_type.value:
                case StoresTypes.CHROMA.value:
                    try:
                        from mindcraft.infra.vectorstore.chroma import Chroma
                    except ImportError:
                        raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                        f"`pip install chromadb`")

                    cls._instance._store = Chroma(cls._instance._world_data_path,
                                                  cls._instance._world_name,
                                                  cls._instance._embeddings)
                case _:
                    raise NotImplementedError(f"{kwargs.get('store_type')} not implemented")

            if cls._instance._remote:
                print("Client for the Remote server configured. Please start your server running:\n"
                      f"`python -m vllm.entrypoints.openai.api_server "
                      f"--model \"{cls._instance._llm_type.value['name']}\" --trust-remote-code &`")
                print(f"Mindcraft will try to reach out this server:\n{FAST_INFERENCE_URL}\n")
                print(f"If that's not the right HOST/PORT, overwrite them setting env vars `MINDCRAFT_HOST` and "
                      f"`MINDCRAFT_PORT`.")

        return cls._instance

    @property
    def embeddings(self):
        """ Getter for the embeddings property"""
        if self._instance is None:
            return None
        return self._instance._embeddings

    @embeddings.setter
    def embeddings(self, value: EmbeddingsTypes):
        """ Setter for the embeddings property"""
        if self._instance is None:
            return
        self._instance._embeddings = value

    @property
    def llm_type(self):
        """ Getter for the llm_type property"""
        if self._instance is None:
            return None
        return self._instance._llm_type

    @llm_type.setter
    def llm_type(self, value: LLMType):
        """ Setter for the llm_type property"""
        if self._instance is None:
            return
        self._instance._llm_type = value

    @property
    def llm(self):
        """ Getter for the llm_type property"""
        if self._instance is None:
            return None
        return self._instance._llm

    @llm.setter
    def llm(self, value: LLM):
        """ Setter for the llm_type property"""
        if self._instance is None:
            return
        self._instance._llm = value

    @property
    def npcs(self):
        """ Getter for the npcs property"""
        if self._instance is None:
            return None
        return self._instance._npcs

    @npcs.setter
    def npcs(self, value: dict):
        """ Setter for the npcs property"""
        if self._instance is None:
            return
        self._instance._npcs = value

    @property
    def fast(self):
        """ Getter for the fast property"""
        if self._instance is None:
            return None
        return self._instance._fast

    @fast.setter
    def fast(self, value: bool):
        """ Setter for the fast property"""
        if self._instance is None:
            return
        self._instance._fast = value

    @property
    def remote(self):
        """ Getter for the remote property"""
        if self._instance is None:
            return None
        return self._instance._remote

    @remote.setter
    def remote(self, value: bool):
        """ Setter for the remote property"""
        if self._instance is None:
            return
        self._instance._remote = value

    @property
    def streaming(self):
        """ Getter for the streaming property"""
        if self._instance is None:
            return None
        return self._instance._streaming

    @streaming.setter
    def streaming(self, value: bool):
        """ Setter for the streaming property"""
        if self._instance is None:
            return
        self._instance._streaming = value

    @property
    def world_name(self):
        """ Getter for the world_name property"""
        if self._instance is None:
            return None
        return self._instance._world_name

    @world_name.setter
    def world_name(self, value: str):
        """ Setter for the world_name property"""
        if self._instance is None:
            return
        self._instance._world_name = value

    @property
    def store(self):
        """ Getter for the store property"""
        if self._instance is None:
            return None
        return self._instance._store

    @store.setter
    def store(self, value: Store):
        """ Setter for the store property"""
        if self._instance is None:
            return
        self._instance._store = value

    @property
    def store_type(self):
        """ Getter for the store_type property"""
        if self._instance is None:
            return None
        return self._instance._store_type

    @store_type.setter
    def store_type(self, value: Store):
        """ Setter for the store_type property"""
        if self._instance is None:
            return
        self._instance._store_type = value

    @classmethod
    def is_created(cls) -> bool:
        """:return Returns true if the Singleton instance of the World is already created. False otherwise"""
        return cls._instance is not None

    @classmethod
    def get_lore(cls,
                 topic: str,
                 num_results: int = 5,
                 known_by: str = None,
                 exact_match: str = None,
                 min_similarity: float = 0.85) -> SearchResult:
        """
        Gets the lore from the world relevant to a topic, and filtered by who knows about it (known_by). You can use
        `num_results` to get the top-n results and `exact_match` if you want the results to include something literal.
        :param topic: the topic you are looking for in the Vector Store
        :param num_results: the max. number of results to retrieve
        :param known_by: filters by who know about this piece of lore. By default, (None) will look for commonly known
        by all NPCs.
        :param exact_match: Only returns documents which include literal expressions
        :param min_similarity: The minimum similarity the document should have compared to the topic
        :return SearchResult
        """

        all_known_by = [settings.ALL]
        if known_by is not None and known_by != settings.ALL:
            all_known_by.append(known_by)

        return cls._instance.store.query(
            topic,
            num_results,
            all_known_by,
            exact_match,
            min_similarity)

    @classmethod
    def add_lore(cls,
                 lore_text: str,
                 lore_id: str,
                 known_by: list[str]):
        """
            Stores a piece of lore which happened in a world.
        :param lore_text: chronicle to be stored
        :param lore_id: the id of the piece of lore
        :param known_by: list of character_ids who know the chronicle
        """
        logger.info(f"Processing {lore_id} [{lore_text[:10]}...]")
        cls._instance.store.add_to_collection(
            text=lore_text,
            metadata={"known_by": SEPARATOR.join(known_by)},
            text_id=lore_id
        )

    @classmethod
    def book_to_world(
            cls,
            book_path: str,
            text_splitter: TextSplitterTypes,
            max_units: int,
            overlap: int,
            known_by: list[str] = None,
            encoding='utf-8'):
        """
        Reads a file describing a world (a book, for example). Splits the text into small chunks and stores them
        in the world. You can use any of the text splitters available in TextSplitterTypes.
        :param book_path: the path to the book
        :param text_splitter: one of those available in TextSplitterTypes (TokenTextSplitter, SentenceTextSplitter...)
        :param known_by: known by characters. If None, `all` will be included
        :param overlap: number of units (tokens, sentences) to overlap with previous/next chunks
        :param max_units: number of units (tokens, sentences) to accumulate in a chunk
        :param encoding: encoding of the books
        """
        with open(book_path, 'r', encoding=encoding) as f:
            book = f.read()

            match text_splitter:
                case TextSplitterTypes.MAX_TOKENS_SPLITTER:
                    text_splitter = TokenTextSplitter(
                        overlap=overlap,
                        max_units=max_units
                    )
                case TextSplitterTypes.SENTENCE_SPLITTER:
                    text_splitter = SentenceTextSplitter(
                        overlap=overlap,
                        max_units=max_units
                    )
                case _:
                    raise NotImplementedError(f"{str(text_splitter)} not implemented")

            loading = ['|', '/', '-', '\\']
            for i, chunk in enumerate(text_splitter.split_text(book)):
                print(f"\r{loading[i % len(loading)]}", end="")
                cls.add_lore(chunk,
                             str(i),
                             known_by if known_by is not None else [ALL])
            print()

    @classmethod
    def retrieve_answer_from_llm(cls,
                                 prompt: str,
                                 max_tokens: int = 100,
                                 do_sample: bool = True,
                                 temperature: float = 0.8) -> Union[Iterator[str], str]:
        """
        Sends a prompt to the LLM. You can specify the max. number of tokens to retrieve and if you do sampling when
        generating the text.
        :param prompt: the prompt to use
        :param max_tokens: max tokens to receive
        :param do_sample: apply stochastic selection of tokens to prevent always generating the same wording.
        :param temperature: temperature or how creative the answer should be
        :return: an iterator to the text of the answer (streaming=True) or the answer (streaming=False)
        """
        if cls._instance.fast:
            if cls._instance.llm is None:
                if cls._instance.remote:
                    cls._instance.llm = RemoteVLLM(cls._instance.llm_type, temperature)
                else:
                    cls._instance.llm = LocalVLLM(cls._instance.llm_type, temperature)
        else:
            if cls._instance.llm is None:
                cls._instance.llm = LocalLLM(cls._instance.llm_type, temperature)

        for chunk in cls._instance.llm.retrieve_answer(prompt,
                                                       max_tokens,
                                                       do_sample,
                                                       cls._instance.llm_type.value['template'],
                                                       cls._instance.streaming):
            yield chunk

    @classmethod
    def get_instance(cls):
        """ Returns the Singleton instance of the World"""
        return cls._instance

    @classmethod
    def delete_collection(cls):
        """
        Deletes a collection from the Vector Store
        """
        match cls._instance.store_type.value:
            case StoresTypes.CHROMA.value:
                try:
                    from mindcraft.infra.vectorstore.chroma import Chroma
                except ImportError:
                    raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                    f"`pip install chromadb`")
                cls._instance.store.delete_collection()
            case _:
                raise NotImplementedError(f"{cls._instance.store_type} not implemented")

    @classmethod
    def create_prompt(cls,
                      memories: list[str],
                      world_knowledge: list[str],
                      character_name: str,
                      topic: str,
                      personalities: list[str],
                      motivations: list[str],
                      conversational_style: list[str],
                      mood: str = None) -> str:
        """
        Static method that creates the prompt to send to the LLM, gathering all the information from the world,
        past interactions, personalities, motivation, mood, conversational styles, etc.
        :param memories: A list of past interactions with a specific character about this topic
        :param world_knowledge: Pieces of lore/knowledge in the world about this topic
        :param character_name: The name of the character
        :param topic: The topic you are asking about
        :param personalities: A list of personalities of the NPC who is answering. For example: `wise`, `intelligent`
        :param motivations: A list of motivations seeked by the NPC who is answering. For example:
        `protecting the nature`
        :param conversational_style: A list of examples of a conversation which happened when the NPC was in a similar
        mood
        :param mood: The current mood of the NPC
        :return: the prompt
        """

        return Prompt.create(memories,
                             world_knowledge,
                             character_name,
                             cls._instance.world_name,
                             topic,
                             personalities,
                             motivations,
                             conversational_style,
                             mood,
                             prompt_template=cls._instance.llm_type)
