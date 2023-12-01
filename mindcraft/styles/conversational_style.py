from mindcraft.features.mood import Mood
from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from settings import STYLES_DATA_PATH


class ConversationalStyle:
    def __init__(self,
                 store_type: StoresTypes,
                 character_id: str,
                 styles_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        Long-term memory. It stores everything that happened to a character.
        They are kept in the vector store, so the retrieval is slower than the STM.
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
            Stores an example conversation into the vector store.
        :param text: last interaction happened to store in LTM.
        :param mood:
        """
        self.store.add_to_collection(
            text=text,
            metadata={'mood': mood.feature},
            text_id=str(self.store.count()))

    def retrieve_interaction_by_mood(self,
                                     mood: str) -> dict:
        """

        :param mood:
        :return:
        """
        return self.store.get(where={'mood': mood})
