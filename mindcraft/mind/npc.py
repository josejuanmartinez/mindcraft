from mindcraft.infra.embeddings.embeddings import Embeddings
from mindcraft.memory.ltm import LTM
from mindcraft.memory.stm import STM


class NPC:
    def __init__(self,
                 character_id: str,
                 stm_capacity: int = 15,
                 ltm_embeddings: Embeddings = Embeddings.MINILM):
        """
        A class managing the Non-player Character, including short-term, long-term memory, backgrounds, motivations
        to create the answer.
        :param character_id: the unique id of the character
        :param stm_capacity: the short-term memory capacity. STM stores in memory for fast retrieval this number
        of interactions
        """
        self.ltm = LTM(character_id, ltm_embeddings)
        self.stm = STM(self.ltm, stm_capacity)

