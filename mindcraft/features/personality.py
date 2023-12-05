class Personality:
    def __init__(self, feature: str):
        """
        Class that defines a permanent personality feature of a NPC. If you are looking for a feature that can change
         over the time, use `Mood` instead
        :param feature: the name of the personality feature, for example, `wise`.
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


