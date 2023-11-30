from mindcraft.infra.splitters.sentence_text_splitter import SentenceTextSplitter
from mindcraft.infra.splitters.token_text_splitter import TokenTextSplitter
from mindcraft.infra.splitters.text_splitters_types import TextSplitterTypes
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.vectorstore.store import Store
from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.settings import SEPARATOR, LOGGER_FORMAT, WORLD_DATA_PATH, ALL

import logging

logging.basicConfig(format=LOGGER_FORMAT, datefmt='%d-%m-%Y:%H:%M:%S', level=logging.ERROR)
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
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        :param llm_type: Embeddings to use in LTM in the VectorS Store.
        """
        if 'world_name' not in kwargs:
            raise Exception("To instantiate a world, please add the name of the world in `world_name`")

        if 'store_type' not in kwargs:
            raise Exception("`store_type` not found in World() initializer")

        if 'embeddings' not in kwargs:
            logger.warning("`embeddings` not found in World() initializer. "
                           f"Initializing to {str(EmbeddingsTypes.MINILM.value)}")

        if 'llm_type' not in kwargs:
            logger.warning(f"`llm_type` not found in World() initializer. Initializing to {LLMType.ZEPHYR7B}")

        create_world = False
        destroying_world = False

        if cls._instance is None:
            create_world = True
        elif kwargs.get('world_name') != cls._instance.world_name:
            create_world = True
            destroying_world = True

        if create_world:
            if destroying_world:
                logger.info(f"Changing world from {cls._instance.world_name} to {kwargs.get('world_name')}")

            cls._instance = super().__new__(cls)
            cls._instance._world_name = kwargs.get('world_name')
            cls._instance._embeddings = kwargs.get('embeddings') if 'embeddings' in kwargs else EmbeddingsTypes.MINILM
            cls._instance._store_type = kwargs.get('store_type')
            cls._instance._llm_type = kwargs.get('llm_type') if 'llm_type' in kwargs else LLMType.ZEPHYR7B
            cls._instance._llm = LLM(cls._instance._llm_type)

            match cls._instance._store_type.value:
                case StoresTypes.CHROMA.value:
                    try:
                        from mindcraft.infra.vectorstore.chroma import Chroma
                    except ImportError:
                        raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                        f"`pip install chromadb`")

                    cls._instance._store = Chroma(WORLD_DATA_PATH,
                                                  cls._instance._world_name,
                                                  cls._instance._embeddings)
                case _:
                    raise NotImplementedError(f"{kwargs.get('store_type')} not implemented")

        return cls._instance

    @property
    def embeddings(self):
        if self._instance is None:
            return None
        return self._instance._embeddings

    @embeddings.setter
    def embeddings(self, value: EmbeddingsTypes):
        if self._instance is None:
            return
        self._instance._embeddings = value

    @property
    def llm(self):
        if self._instance is None:
            return None
        return self._instance._llm

    @llm.setter
    def llm(self, value: LLMType):
        if self._instance is None:
            return
        self._instance._llm = value

    @property
    def world_name(self):
        if self._instance is None:
            return None
        return self._instance._world_name

    @world_name.setter
    def world_name(self, value: str):
        if self._instance is None:
            return
        self._instance._world_name = value

    @property
    def store(self):
        if self._instance is None:
            return None
        return self._instance._store

    @store.setter
    def store(self, value: Store):
        if self._instance is None:
            return
        self._instance._store = value

    @classmethod
    def is_created(cls):
        return cls._instance is not None

    @classmethod
    def get_lore(cls,
                 topic: str,
                 num_results: int = 5,
                 known_by: str = None) -> dict:
        """

        :param topic:
        :param num_results:
        :param known_by:
        :return:
        """

        where = {}
        if known_by is not None and known_by != 'all':
            where = {"known_by": known_by}

        return cls._instance.store.query(topic, num_results, where)

    @classmethod
    def add_lore(cls,
                 lore_text: str,
                 lore_id: str,
                 known_by: list[str]):
        """
            Chronicles (stores) an event happened in a world.
        :param lore_text: chronicle to be stored
        :param lore_id:
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
            known_by: list[str] = None):
        """
        Reads a file describing a world (a book, for example). Splits the text into small chunks and stores them
        in the world. You can use any of the text splitters available in TextSplitterTypes.
        :param book_path: the path to the book
        :param text_splitter: one of those avialable in TextSplitterTypes (TokenTextSplitter, SentenceTextSplitter...)
        :param known_by: known by characters. If None, `all` will be included
        :param overlap: number of units (tokens, sentences) to overlap with previous/next chunks
        :param max_units: number of units (tokens, sentences) to accumulate in a chunk

        :return:
        """
        with open(book_path, 'r') as f:
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

            for i, chunk in enumerate(text_splitter.split_text(book)):
                print(".", end="")
                # print(chunk)
                cls.add_lore(chunk,
                             str(i),
                             known_by if known_by is not None else [ALL])
            print()

    @classmethod
    def get_instance(cls):
        return cls._instance
