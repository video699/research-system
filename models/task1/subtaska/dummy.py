"""
    This module implements unsupervised task 1, subtask A dummy models that provide baseline
    scores and integration testing of the evaluation code.
"""

from random import random, randint

from .base import Model

class Best(Model):
    """
        This class represents a task 1, subtask A model that cheats to obtain the best
        possible results.
    """
    def predict(self, observations):
        rankings = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                ranking = [(1.0 if page in screen.matching_pages else 0.0) for page in pages]
                rankings.append(ranking)
        return rankings

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Best)"

class Worst(Model):
    """
        This class represents a task 1, subtask A model that cheats to obtain the worst
        possible results.
    """
    def predict(self, observations):
        rankings = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                ranking = [(0.0 if page in screen.matching_pages else 1.0) for page in pages]
                rankings.append(ranking)
        return rankings

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Worst)"

class Random(Model):
    """
        This class represents a task 1, subtask A model that picks results at random.
    """
    def predict(self, observations):
        rankings = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                ranking = [random() for page in pages]
                rankings.append(ranking)
        return rankings

    def _filename(self):
        return "%s.%s" % (__name__, self.__class__.__name__)

    def __repr__(self):
        return "(Random)"

BEST = Best()
WORST = Worst()
RANDOM = Random()
