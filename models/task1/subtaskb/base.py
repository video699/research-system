"""
    This module defines the basic mixin implemented by task 1, subtask B models.
"""

from ..base import Model as BaseModel

class Model(BaseModel):
    """ This mixin represents a task 1, subtask B model."""
    def predict(self, observations):
        """Predicts whether the screens in a provided video display any of the document pages in
        another provided video and returns the predictions as a list of truthy and falsy values.

        Parameters:
            observations    The provided list of 2-tuples, where each tuple consists of a video
                            containing screens and a video containing document pages.
        """
        raise NotImplementedError()

    def _filename(self):
        raise NotImplementedError()
