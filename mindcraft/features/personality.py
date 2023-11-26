class Personality:
    def __init__(self, feature: str):
        """

        :param feature:
        """
        self._feature = feature

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, value: str):
        self._feature = value


