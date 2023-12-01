from typing import Optional

from mindcraft.infra.embeddings.embeddings_types import EmbeddingsTypes


class Store:
    def __init__(self, path: str, collection_name: str, embeddings: EmbeddingsTypes):
        """
        :param path:
        :param collection_name:
        """
        self.collection_name = collection_name
        self.path = f"{path}/{collection_name}"
        self.embeddings = embeddings

    def instantiate_client(self):
        """

        :return:
        """
        raise NotImplementedError()

    def get_embeddings(self, text):
        """

        :param text:
        :return:
        """
        raise NotImplementedError()

    def query(self, text: str, num_result: int, all_known_by: dict, exact_match: str = None):
        """

        :param exact_match:
        :param text:
        :param num_result:
        :param all_known_by:
        :return:
        """
        raise NotImplementedError()

    def add_to_collection(self,
                          text_id: str,
                          text: str,
                          metadata: Optional[dict]):
        """
        :param text_id:
        :param text:
        :param metadata:
        :return:
        """
        raise NotImplementedError()

    def count(self) -> int:
        """

        :return:
        """
        raise NotImplementedError()
