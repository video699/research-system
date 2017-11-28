"""
    This module implements task 1, subtask A and B dummy models that provide baseline scores and
    integration testing of the evaluation code.
"""

from random import random, randint

from .base import Task1SubtaskAModel, Task1SubtaskBModel

class BestTask1SubtaskAModel(Task1SubtaskAModel):
    """
        This class represents a task 1, subtask A model that cheats to obtain the best
        possible results.
    """
    def predict(self, observations):
        for screen, pages in observations:
            ranking = [(1.0 if page in screen.matching_pages else 0.0) for page in pages]
            yield ranking

    def _filename(self):
        return "%s.%s.%s" % (super(BestTask1SubtaskAModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Best)"

class BestTask1SubtaskBModel(Task1SubtaskBModel):
    """
        This class represents a task 1, subtask B model that cheats to obtain the best
        possible results.
    """
    def predict(self, observations):
        for screen, pages in observations:
            yield any(page in screen.matching_pages for page in pages)

    def _filename(self):
        return "%s.%s.%s" % (super(BestTask1SubtaskBModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Best)"

class WorstTask1SubtaskAModel(Task1SubtaskAModel):
    """
        This class represents a task 1, subtask A model that cheats to obtain the worst
        possible results.
    """
    def predict(self, observations):
        for screen, pages in observations:
            ranking = [(0.0 if page in screen.matching_pages else 1.0) for page in pages]
            yield ranking

    def _filename(self):
        return "%s.%s.%s" % (super(WorstTask1SubtaskAModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Worst)"

class WorstTask1SubtaskBModel(Task1SubtaskBModel):
    """
        This class represents a task 1, subtask B model that cheats to obtain the worst
        possible results.
    """
    def predict(self, observations):
        for screen, pages in observations:
            yield all(page not in screen.matching_pages for page in pages)

    def _filename(self):
        return "%s.%s.%s" % (super(WorstTask1SubtaskBModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Worst)"

class RandomTask1SubtaskAModel(Task1SubtaskAModel):
    """
        This class represents a task 1, subtask A model that picks results at random.
    """
    def predict(self, observations):
        for _, pages in observations:
            ranking = [(random(), page) for page in pages]
            yield ranking

    def _filename(self):
        return "%s.%s.%s" % (super(RandomTask1SubtaskAModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Random)"

class RandomTask1SubtaskBModel(Task1SubtaskBModel):
    """
        This class represents a task 1, subtask B model that picks results at random.
    """
    def predict(self, observations):
        for _, __ in observations:
            yield randint(0, 1)

    def _filename(self):
        return "%s.%s.%s" % (super(RandomTask1SubtaskBModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Random)"

class ConservativeModel(Task1SubtaskBModel):
    """
        This class represents a task 1, subtask B model that marks all screens as matchable,
        since wrongly marking a screen as non-matchable is costly in terms of the evaluation metric.
    """
    def predict(self, observations):
        for _, __ in observations:
            yield True

    def _filename(self):
        return "%s.%s.%s" % (super(ConservativeModel, self)._filename(), __name__, \
                             self.__class__.__name__)

    def __repr__(self):
        return "(Conservative)"

BEST_TASK1_SUBTASKA = BestTask1SubtaskAModel()
BEST_TASK1_SUBTASKB = BestTask1SubtaskBModel()
WORST_TASK1_SUBTASKA = WorstTask1SubtaskAModel()
WORST_TASK1_SUBTASKB = WorstTask1SubtaskBModel()
RANDOM_TASK1_SUBTASKA = RandomTask1SubtaskAModel()
RANDOM_TASK1_SUBTASKB = RandomTask1SubtaskBModel()
CONSERVATIVE = ConservativeModel()
