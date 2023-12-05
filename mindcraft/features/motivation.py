class Motivation:
    def __init__(self, feature: str = None):
        """
        Class that defines the motivations of a NPC.
        :param feature: the description of the motivation, for example, `Seeking the destruction of the all living`.
        """
        self._feature = feature

    @property
    def feature(self):
        """
        Getter of the `feature` property
        :return: string
        """
        return self._feature

    @feature.setter
    def feature(self, value: str):
        """
        Setter of the `feature` property
        :param value: string of the feature.
        """
        self._feature = value
