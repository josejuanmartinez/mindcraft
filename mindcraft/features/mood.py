class Mood:
    DEFAULT = 'default'

    def __init__(self, feature: str = None):
        """
        Class that defines the current mood of a NPC. Moods can change over the time. If you are looking for something
        permanent, use `Personality` instead.
        :param feature: the name of the mood, for example, `angry`.
        """
        self._feature = feature if feature is not None else self.DEFAULT

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


