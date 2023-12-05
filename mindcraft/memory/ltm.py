from mindcraft.infra.vectorstore.search_results import SearchResult
from mindcraft.features.mood import Mood
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from settings import LTM_DATA_PATH, ALL


class LTM:
    def __init__(self,
                 store_type: StoresTypes,
                 character_name: str,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        Long-term memory. It stores everything that happened to a character.
        They are kept in the vector store, so the retrieval is slower than the STM.
        :param character_name: the unique `id` of the character
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        match store_type.value:
            case StoresTypes.CHROMA.value:
                try:
                    from mindcraft.infra.vectorstore.chroma import Chroma
                except ImportError:
                    raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                    f"`pip install chromadb`")

                self._store = Chroma(LTM_DATA_PATH, character_name, ltm_embeddings)

            case _:
                raise NotImplementedError(f"{store_type} not implemented")

        self._embeddings = ltm_embeddings
        self._character_id = character_name

    def memorize(self, text: str, mood: Mood):
        """
            Stores a memory or interaction into the vector store, all along with the actual moods which produced it.
        :param text: last interaction happened to store in LTM.
        :param mood: current Mood of the character
        """
        self._store.add_to_collection(
            text=text,
            metadata={'mood': mood.feature if mood is not None else Mood.DEFAULT},
            text_id=str(self._store.count()))

    def remember_about(self,
                       topic: str,
                       num_results: int = 3,
                       min_similarity: float = 0.85) -> SearchResult:
        """
        Retrieves memories from LTM of a character concerning a specific topic.
        :param topic: Topic to remember about
        :param num_results: Max. num of results
        :param min_similarity: min. similarity to filter out irrelevant memories
        """
        return self._store.query(
                text=topic,
                num_results=num_results,
                known_by=[ALL, self._character_id],
                min_similarity=min_similarity
        )
