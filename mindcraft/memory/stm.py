import collections

from mindcraft.memory.ltm import LTM


class STM:
    def __init__(self, ltm: LTM, capacity: int = 15):
        """
        Short-term memory. It stores the last `capacity` interactions and remembers them well.
        They are kept in memory for fast retrieval. After shutting down, they are pushed to long-term memory.
        It's managed by a FIFO dequeue with a `capacity` length.
        :param ltm: long-term memory object, to store old elements in short-term
        :param capacity: number of past interactions stored in the short-term memory.
        """
        self.ltm = ltm
        self.capacity = capacity
        self.interactions = collections.deque(maxlen=self.capacity)

    def remember(self, text: str):
        """
            Pops the oldest element in the STM, and sends it to LTM.
            Then, adds the newest interaction.
        :param text: last interaction happened to store in STM.
        """
        if len(self.interactions) == self.capacity:
            self.ltm.remember(self.interactions[0])

        self.interactions.append(text)
