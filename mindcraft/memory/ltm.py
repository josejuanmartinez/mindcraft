from chromadb.utils import embedding_functions

from mindcraft.infra.embeddings.embeddings import Embeddings
from mindcraft.infra.vectorstore.chroma import Chroma


class LTM:
    def __init__(self, character_id: str, ltm_embeddings: Embeddings = Embeddings.MINILM):
        """
        Long-term memory. It stores everything that happened to a character.
        They are kept in the vector store, so the retrieval is slower than the STM.
        :param character_id: the unique `id` of the character
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        self.store = Chroma(character_id)
        self.embeddings = ltm_embeddings

    def remember(self, text: str):
        """
            Stores a memory or interaction into the vector store.
        :param text: last interaction happened to store in LTM.
        """
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))
        self.store.add_to_collection(
            text=[text],
            text_embeddings=sentence_transformer_ef([text])
        )
