from chromadb.utils import embedding_functions

from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.vectorstore.chroma import Chroma


class LTM:
    def __init__(self, character_id: str, ltm_embeddings: EmbeddingsTypes = EmbeddingsTypes.MINILM):
        """
        Long-term memory. It stores everything that happened to a character.
        They are kept in the vector store, so the retrieval is slower than the STM.
        :param character_id: the unique `id` of the character
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        self.store = Chroma(character_id)
        self.items = self.store.count()
        self.embeddings = ltm_embeddings

    def memorize(self, text: str):
        """
            Stores a memory or interaction into the vector store.
        :param text: last interaction happened to store in LTM.
        """
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))
        self.store.add_to_collection(
            text=text,
            text_embeddings=sentence_transformer_ef([text]),
            metadata=None,
            text_id=str(self.items))
        self.items += 1

    def remember_about(self,
                       topic: str,
                       n_results: int = 3) -> dict:
        """

        :param topic:
        :param n_results:
        :return:
        """
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))

        return self.store.collection.query(
                query_embeddings=sentence_transformer_ef([topic]),
                n_results=n_results)
