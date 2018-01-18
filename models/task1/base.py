"""
    This module defines the basic mixin implemented by task 1 models.
"""

class Model(object):
    def fit(self, videos):
        """Trains the model using the provided videos.

        Parameters:
            videos          The provided list of videos.
        """
        pass

    def _filename(self):
        """Returns the filename that serves as a basename to store data related to this model."""
        raise NotImplementedError()
