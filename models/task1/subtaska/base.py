"""
    This module defines the basic mixin implemented by task 1, subtask A models.
"""

from ..base import Model as BaseModel

class Model(BaseModel):
    """ This mixin represents a task 1, subtask A model."""
    def predict(self, observations):
        """Predicts the rankings of the document pages in a provided video based on their similarity
        to the screens in another provided video and returns the predicted rankings as a list.

        Parameters:
            observations    The provided list of 2-tuples, where each tuple consists of a video
                            containing screens and a video containing document pages.
        """
        raise NotImplementedError()

    def _filename(self):
        raise NotImplementedError()
