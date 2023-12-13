from mindcraft.infra.vectorstore.search_results import SearchResult
from mindcraft.features.mood import Mood
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.settings import STYLES_DATA_PATH


class ConversationalStyle:
    def __init__(self,
                 store_type: StoresTypes,
                 character_id: str,
                 styles_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        Class that stores how characters speak depending on their moods.
        They are kept in the vector store
        :param store_type: type of vector store from those available in StoresTypes
        :param character_id: the unique `id` of the character
        :param styles_embeddings: Embeddings to use in the conversations in the Vector Store.
        """
        match store_type.value:
            case StoresTypes.CHROMA.value:
                try:
                    from mindcraft.infra.vectorstore.chroma import Chroma
                except ImportError:
                    raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                    f"`pip install chromadb`")

                self.store = Chroma(STYLES_DATA_PATH, character_id, styles_embeddings)

            case _:
                raise NotImplementedError(f"{store_type} not implemented")

        self.embeddings = styles_embeddings

    def memorize(self, text: str, mood: Mood):
        """
            Stores an example conversation of a character for a specific mood into the vector store.
        :param text: last interaction happened to store in LTM.
        :param mood: the mood the npc had when said this
        """
        self.store.add_to_collection(
            text=text,
            metadata={'mood': mood.feature if mood is not None else Mood.DEFAULT},
            text_id=str(self.store.count()))

    def retrieve_interaction_by_mood(self,
                                     mood: str) -> SearchResult:
        """
        Retrieves examples of interactions for a specific mood
        :param mood: the current mood of the character
        :return SearchResult
        """
        return self.store.get(where={'mood': mood})
