from mindcraft.infra.vectorstore.stores_types import StoresTypes
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from settings import LTM_DATA_PATH


class LTM:
    def __init__(self,
                 store_type: StoresTypes,
                 character_id: str,
                 ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        Long-term memory. It stores everything that happened to a character.
        They are kept in the vector store, so the retrieval is slower than the STM.
        :param character_id: the unique `id` of the character
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        match store_type.value:
            case StoresTypes.CHROMA.value:
                try:
                    from mindcraft.infra.vectorstore.chroma import Chroma
                except ImportError:
                    raise Exception(f"To use `chromadb` as your vector store, please install it first using pip:\n"
                                    f"`pip install chromadb`")

                self.store = Chroma(LTM_DATA_PATH, character_id, ltm_embeddings)

            case _:
                raise NotImplementedError(f"{store_type} not implemented")

        self.embeddings = ltm_embeddings

    def memorize(self, text: str):
        """
            Stores a memory or interaction into the vector store.
        :param text: last interaction happened to store in LTM.
        """
        self.store.add_to_collection(
            text=text,
            metadata=None,
            text_id=str(self.store.count()))

    def remember_about(self,
                       topic: str,
                       n_results: int = 3) -> dict:
        """

        :param topic:
        :param n_results:
        :return:
        """
        return self.store.query(
                text=topic,
                num_results=n_results,
                where={}
        )
