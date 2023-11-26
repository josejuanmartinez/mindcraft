from typing import Optional

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

    def add_to_collection(self,
                          text: str,
                          text_embeddings: Embeddings,
                          metadata: Optional[dict],
                          text_id: str):
        """

        :param text:
        :param text_embeddings:
        :param metadata:
        :param text_id: 
        :return:
        """
        self.collection.add(
            documents=[text],
            embeddings=text_embeddings,
            metadatas=[metadata] if metadata is not None else [],
            ids=[text_id]
        )

    def count(self) -> int:
        """

        :return:
        """
        if self.collection is None:
            return 0
        return self.collection.count()
