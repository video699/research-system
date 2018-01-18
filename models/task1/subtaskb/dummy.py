"""
    This module implements unsupervised task 1, subtask B dummy models that provide baseline
    scores and integration testing of the evaluation code.
"""

from random import random, randint

from .base import Model

class Best(Model):
    """
        This class represents a task 1, subtask B model that cheats to obtain the best
        possible results.
    """
    def predict(self, observations):
        predictions = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                prediction = any(page in screen.matching_pages for page in pages)
                predictions.append(prediction)
        return predictions

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Best)"

class Worst(Model):
    """
        This class represents a task 1, subtask B model that cheats to obtain the worst
        possible results.
    """
    def predict(self, observations):
        predictions = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                prediction = all(page not in screen.matching_pages for page in pages)
                predictions.append(prediction)
        return predictions

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Worst)"

class Random(Model):
    """
        This class represents a task 1, subtask B model that picks results at random.
    """
    def predict(self, observations):
        predictions = []
        for screen_video, _ in observations:
            for __ in screen_video.screens:
                prediction = randint(0, 1)
                predictions.append(prediction)
        return predictions

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Random)"

class Conservative(Model):
    """
        This class represents a task 1, subtask B model that marks all screens as matchable,
        since wrongly marking a screen as non-matchable is costly in terms of the evaluation metric.
    """
    def predict(self, observations):
        predictions = []
        for screen_video, _ in observations:
            for __ in screen_video.screens:
                prediction = True
                predictions.append(prediction)
        return predictions

    def _filename(self):
        return "%s.%s" % (__name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Conservative)"

BEST = Best()
WORST = Worst()
RANDOM = Random()
CONSERVATIVE = Conservative()
