from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from mindcraft.infra.vectorstore.search_results import SearchResult
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes
from mindcraft.infra.vectorstore.store import Store


class Chroma(Store):
    def __init__(self, path: str, collection_name: str, embeddings: EmbeddingsTypes):
        """
        ChromaDB manager, in charge of loading, persisting and query the character memories and interactions
        :param collection_name: name of the collection
        :param path: path where to save in disk the collection
        :param embeddings: type of EmbeddingsType
        """
        super().__init__(path, collection_name, embeddings)
        self.client = self.instantiate_client()
        self.collection = self.create_or_get_collection()

    def instantiate_client(self):
        """
        Creation of the ChromaDB client persisting data to disk.
        """
        return chromadb.PersistentClient(path=self.path)

    def create_or_get_collection(self):
        """
        ChromaDB abstraction to retrieve a collection if already exists, or create it otherwise.
        """
        return self.client.get_or_create_collection(name=self.collection_name)

    def get_embeddings(self, text):
        """
        Transforms text to embeddings using the EmbeddingsType you selected in the constructor
        :param text: Text to calculate embeddings from
        """
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(self.embeddings.value))
        return sentence_transformer_ef([text])

    def add_to_collection(self,
                          text: str,
                          metadata: Optional[dict],
                          text_id: str):
        """
        Adds a text to a ChromaDB collection, after calculating its embeddings.
        It accepts metadata as well to filter the results during retrieval.
        :param text: Text to be transformed into embeddings and stored.
        :param metadata: Dictionary with any key:value pair you want to store, e.g `known_by`: `galadriel`
        :param text_id: A unique id of the text
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
        Counts the number of items in a collection
        :return: integer with the number of items
        """
        if self.collection is None:
            return 0
        return self.collection.count()

    def query(self,
              text: str,
              num_results: int,
              known_by: list,
              exact_match: str = None,
              min_similarity: float = 0.85) -> SearchResult:
        """
        Implementation of ChromaDB of the retrieval function
        :param text: Text to retrieve similar entries from
        :param num_results: Max. number of results
        :param known_by: Filter the entries in the vector store by the character's id. use settings.`ALL` for pieces of
        lore known to everyone
        :param exact_match: Filter the entries by a text you want to appear explicitly
        :param min_similarity: The minimum similarity the document should have compared to the topic
        :return: SearchResults class
        """
        if len(known_by) == 1:
            where = {"known_by": known_by[0]}
        else:
            where = {
                "$or": [{"known_by": x} for x in known_by]
            }
        where_document = dict()
        if exact_match is not None:
            where_document = {"$contains": exact_match}
        results = self.collection.query(
            query_embeddings=self.get_embeddings(text),
            n_results=num_results,
            where=where,
            where_document=where_document)

        result = SearchResult()
        if 'distances' in results:
            if len(results['distances']) > 0:
                for i, d in enumerate(results['distances'][0]):
                    # In ChromaDB we retrieve distances not similarities
                    # so we need to check that the distance is smaller than the min. similarity
                    if d <= min_similarity:
                        result.documents.append(results['documents'][0][i])
                        result.distances.append(results['distances'][0][i])
        return result

    def get(self, where: dict) -> SearchResult:
        """
        ChromaDB `get` method that queries a collection using a `where` (dict) clause, that checks metadata.
        :param where: dictionary of key:values to be used when checking metadata to filter the results
        :return SearchResult
        """
        results = self.collection.get(where=where)
        result = SearchResult()
        if 'documents' in results:
            if len(results['documents']) > 0:
                result.documents = results['documents'][0]
        return result
