import chromadb
from chromadb import Embeddings

from mindcraft.settings import DIR_PATH


class Chroma:
    def __init__(self, character_id: str, path: str = DIR_PATH):
        """
        ChromaDB manager, in charge of loading, persisting and query the character memories and interactions
        :param character_id:
        """
        self.client = chromadb.PersistentClient(path=f"{path}/{character_id}")
        self.collection = self.client.get_or_create_collection(name=character_id)

    def add_to_collection(self, text: str, text_embeddings: Embeddings, metadatas: dict):
        """

        :param metadatas:
        :param text:
        :param text_embeddings:
        :return:
        """
        self.collection.add(
            documents=[text],
            embeddings=text_embeddings,
            metadatas=[metadatas],
            ids=[]
        )

