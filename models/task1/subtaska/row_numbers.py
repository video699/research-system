"""
    This module implements a task 1, subtask A model that ranks pairs of images based on the
    difference in their number of text rows.
"""

import logging

from .base import Model
from preprocessing import Image, IMAGE_KWARGS

LOGGER = logging.getLogger(__name__)

class RowNumbers(Model):
    """
        This class represents a task 1, subtask A model that ranks pairs of images based on the
        difference in their number of text rows.
    """

    def __init__(self, kwargs):
        """Constructs an unsupervised task1, subtask A model based on the number of text rows.

        Parameters:
            kwargs  The parameters for the preprocessing of images.
        """
        self.kwargs = kwargs

    def predict(self, observations):
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            pages_num_rows = [Image(page, **self.kwargs).get_rows()[0] for page in pages]
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                screen_num_rows = Image(screen, **self.kwargs).get_rows()[0]
                ranking = [-abs(screen_num_rows-page_num_rows) for page_num_rows in pages_num_rows]
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _filename(self):
        return "%s.%s-%s" % (__name__, self.__class__.__name__, self.kwargs.__repr__())

    def __repr__(self):
        return "Row numbers (%s)" % self.kwargs

ROW_NUMBERS = [RowNumbers(kwargs) for kwargs in IMAGE_KWARGS]
