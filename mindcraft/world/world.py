from chromadb.utils import embedding_functions

from mindcraft.infra.embeddings.embeddings import Embeddings
from mindcraft.infra.splitters.text_splitter import TextSplitter
from mindcraft.infra.vectorstore.chroma import Chroma


class World:
    def __init__(self, world_id: str, ltm_embeddings: Embeddings = Embeddings.MINILM):
        """
        World story. It stores everything that happened in a world.
        They are kept in the vector store.
        Not every NPC will know what happened in the world. Metadata will be used.
        :param world_id: the unique `id` of the character
        :param ltm_embeddings: Embeddings to use in LTM in the VectorS Store.
        """
        self.store = Chroma(world_id)
        self.embeddings = ltm_embeddings

    def add_chronicle(self, chronicle_text: str, known_by: list[str]):
        """
            Chronicles (stores) an event happened in a world.
        :param known_by: list of character_ids who know the chronicle
        :param chronicle_text: chronicle to be stored
        """
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))
        self.store.add_to_collection(
            text=chronicle_text,
            text_embeddings=sentence_transformer_ef([chronicle_text]),
            metadatas={"known_by": known_by}
        )

    def book_to_chronicles(
            self,
            book_path: str,
            known_by: list[str],
            chunk_overlap:int = 25,
            tokens_per_chunk: int = 250):
        """
        Reads a file describing a world (a book, for example). Splits the text into chronicles and stores them
        in the world
        :param known_by:
        :param chunk_overlap:
        :param tokens_per_chunk:
        :param book_path: the path to the book
        :return:
        """
        with open(book_path, 'r') as f:
            book = f.read()

            text_splitter = TextSplitter(
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=tokens_per_chunk,
                encoding_name=str(self.embeddings.value))

            for chunk in text_splitter.split_text(book):
                self.add_chronicle(chunk, known_by)
