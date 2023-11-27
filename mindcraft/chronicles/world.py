from chromadb.utils import embedding_functions

from mindcraft.infra.engine.llm import LLM
from mindcraft.infra.engine.llm_types import LLMType
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.vectorstore.chroma import Chroma
from mindcraft.infra.splitters.text_splitter import TextSplitter
from mindcraft.settings import SEPARATOR, LOGGER_FORMAT

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
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        if 'world_name' not in kwargs:
            raise Exception("To instantiate a world, please add the name of the world in `world_name`")

        create_world = False
        destroying_world = False

        if cls._instance is None:
            create_world = True
        elif 'world_name' in kwargs and kwargs.get('world_name') != cls._instance.world_name:
            create_world = True
            destroying_world = True

        if create_world:
            if destroying_world:
                logger.info(f"Changing world from {cls._instance.world_name} to {kwargs.get('world_name')}")
            cls._instance = super(World, cls).__new__(cls)
            cls._instance._world_name = kwargs.get('world_name')
            cls._instance._store = Chroma(kwargs.get('world_name'))
            cls._instance._embeddings = kwargs.get('embeddings') if 'embeddings' in kwargs else EmbeddingsTypes.MINILM
            cls._instance._llm_type = kwargs.get('llm') if 'llm' in kwargs else LLMType.MISTRAL7B
            cls._instance._llm = LLM(cls._instance._llm_type)

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
    def store(self, value: Chroma):
        # TODO: Abstract class to encapsulate any type of Vector Store
        if self._instance is None:
            return
        self._instance._store = value

    @classmethod
    def is_created(cls):
        return  cls._instance is not None

    @classmethod
    def get_chronicles(cls,
                       topic: str,
                       n_results: int = 5,
                       known_by: str = None) -> dict:
        """

        :param topic:
        :param n_results:
        :param known_by:
        :return:
        """

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(cls._instance.embeddings.value))

        where = {}
        if known_by is not None and known_by != 'all':
            where = {"known_by": known_by}

        return cls._instance.store.collection.query(
                query_embeddings=sentence_transformer_ef([topic]),
                n_results=n_results,
                where=where)

    @classmethod
    def add_chronicle(cls,
                      chronicle_text: str,
                      chronicle_id: str,
                      known_by: list[str]):
        """
            Chronicles (stores) an event happened in a world.
        :param chronicle_text: chronicle to be stored
        :param chronicle_id:
        :param known_by: list of character_ids who know the chronicle
        """
        logger.info(f"Processing {chronicle_id} [{chronicle_text[:10]}...]")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(cls._instance.embeddings.value))
        cls._instance.store.add_to_collection(
            text=chronicle_text,
            text_embeddings=sentence_transformer_ef([chronicle_text]),
            metadata={"known_by": SEPARATOR.join(known_by)},
            text_id=chronicle_id
        )

    @classmethod
    def book_to_chronicles(
            cls,
            book_path: str,
            known_by: list[str] = None,
            chunk_overlap: int = 25,
            tokens_per_chunk: int = 250):
        """
        Reads a file describing a world (a book, for example). Splits the text into chronicles and stores them
        in the world
        :param book_path: the path to the book
        :param known_by:
        :param chunk_overlap:
        :param tokens_per_chunk:
        :return:
        """
        with open(book_path, 'r') as f:
            book = f.read()

            text_splitter = TextSplitter(
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=tokens_per_chunk
            )
            for i, chunk in enumerate(text_splitter.split_text(book)):
                cls.add_chronicle(chunk,
                                  str(i),
                                  known_by if known_by is not None else ['all'])

    @classmethod
    def get_instance(cls):
        return cls._instance
