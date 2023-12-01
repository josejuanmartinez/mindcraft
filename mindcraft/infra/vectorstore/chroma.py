from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.vectorstore.store import Store


class Chroma(Store):
    def __init__(self, path: str, collection_name: str, embeddings: EmbeddingsTypes):
        """
        ChromaDB manager, in charge of loading, persisting and query the character memories and interactions
        :param collection_name:
        :param path:
        :param embeddings:
        """
        super().__init__(path, collection_name, embeddings)
        self.client = self.instantiate_client()
        self.collection = self.create_or_get_collection()

    def instantiate_client(self):
        """

        :return:
        """
        return chromadb.PersistentClient(path=self.path)

    def create_or_get_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name)

    def get_embeddings(self, text):
        """

        :param text:
        :return:
        """

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))
        return sentence_transformer_ef([text])

    def add_to_collection(self,
                          text: str,
                          metadata: Optional[dict],
                          text_id: str):
        """

        :param text:
        :param metadata:
        :param text_id: 
        :return:
        """
        text_embeddings = self.get_embeddings(text)
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

    def query(self, text: str, num_results: int, all_known_by: dict, exact_match: str = None):
        """

        :param text:
        :param num_results:
        :param all_known_by:
        :param exact_match:
        :return:
        """
        where = {
            "$or": [{"known_by": x} for x in all_known_by]
        }
        where_document = dict()
        if exact_match is not None:
            where_document = {"$contains": exact_match}
        return self.collection.query(
            query_embeddings=self.get_embeddings(text),
            n_results=num_results,
            where=where,
            where_document=where_document)

    def get(self, where: dict):
        """

        :param where:
        :return:
        """
        return self.collection.get(where=where)
