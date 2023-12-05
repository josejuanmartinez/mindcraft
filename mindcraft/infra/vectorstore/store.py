from typing import Optional

from mindcraft.infra.vectorstore.search_results import SearchResult
from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes


class Store:
    def __init__(self, path: str, collection_name: str, embeddings: EmbeddingsTypes):
        """
        Abstract manager, in charge of loading, persisting and query the character memories and interactions
        :param collection_name: name of the collection
        :param path: path where to save in disk the collection
        :param embeddings: type of EmbeddingsType
        """
        self.collection_name = collection_name
        self.path = f"{path}/{collection_name}"
        self.embeddings = embeddings

    def instantiate_client(self):
        """
        Creation of the client persisting data to disk.
        """
        raise NotImplementedError()

    def get_embeddings(self, text):
        """
        Transforms text to embeddings using the EmbeddingsType you selected in the constructor
        :param text: Text to calculate embeddings from
        """
        raise NotImplementedError()

    def query(self,
              text: str,
              num_results: int,
              known_by: list,
              exact_match: str = None,
              min_similarity: float = 0.85) -> SearchResult:
        """
        Implementation of the retrieval function
        :param text: Text to retrieve similar entries from
        :param num_results: Max. number of results
        :param known_by: Filter the entries in the vector store by the character's id. use settings.`ALL` for pieces of
        lore known to everyone
        :param exact_match: Filter the entries by a text you want to appear explicitly
        :param min_similarity: The minimum similarity the document should have compared to the topic
        :return SearchResult
        """
        raise NotImplementedError()

    def add_to_collection(self,
                          text_id: str,
                          text: str,
                          metadata: Optional[dict]):
        """
        Adds a text to a collection, after calculating its embeddings.
        It accepts metadata as well to filter the results during retrieval.
        :param text: Text to be transformed into embeddings and stored.
        :param metadata: Dictionary with any key:value pair you want to store, e.g `known_by`: `galadriel`
        :param text_id: A unique id of the text
        """
        raise NotImplementedError()

    def count(self) -> int:
        """
        Counts the number of items in a collection
        :return: integer with the number of items
        """
        raise NotImplementedError()

    def get(self, where: dict) -> SearchResult:
        """
        `get` method that queries a collection using a `where` (dict) clause, that checks metadata.
        :param where: dictionary of key:values to be used when checking metadata to filter the results
        :return SearchResult
        """
        raise NotImplementedError()

