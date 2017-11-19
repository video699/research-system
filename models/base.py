"""
    This module defines the basic mixins implemented by models and the low-level evaluation
    functions invoked by the individual multiprocessing workers.
"""

import random

from numpy import mean, percentile
from sklearn.metrics import zero_one_loss

RANDOM_STATE = 12345
TASK1_SUBTASKB_FALSE_POSITIVE_WEIGHT = 9
TASK1_SUBTASKB_FALSE_NEGATIVE_WEIGHT = 1

def evaluate_task1_subtaska_worker(args):
    """Evaluates the provided task 1, subtask A model and returns the average rank of the matching
    document page.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    random.seed(RANDOM_STATE)
    model, (dataset, split) = args
    average_ranks = []
    for train_index, test_index in split:
        # Perform nested training.
        train_screens = dataset[train_index]
        train_pages = [screen.video.pages for screen in train_screens]
        train_observations = list(zip(train_screens, train_pages))
        train_ground_truth = [[page in screen.matching_pages for page in pages] \
                               for screen, pages in train_observations]
        model.fit(train_observations, train_ground_truth)
        # Perform nested testing.
        test_screens = dataset[test_index]
        test_pages = [screen.video.pages for screen in test_screens]
        test_observations = list(zip(test_screens, test_pages))
        test_ground_truth = [[page in screen.matching_pages for page in pages] \
                              for screen, pages in test_observations]
        predictions = model.predict(test_observations)
        ranks = []
        for relevance_judgements, ranking in zip(test_ground_truth, predictions):
            if not any(relevance_judgements):
                # There is no document page matching the screen.
                continue
            ranked_relevance_judgements, _ = zip(*sorted(zip(relevance_judgements, ranking),
                                                         reverse=True, key=lambda x: x[1]))
            ranks.append(ranked_relevance_judgements.index(True))
        average_ranks.append(mean(ranks))
    return CrossValidationResult(model, average_ranks)

def evaluate_task1_subtaskb_worker(args):
    """Evaluates the provided task 1, subtask B model and returns the average rank of the matching
    document page.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    random.seed(RANDOM_STATE)
    model, (dataset, split) = args
    losses = []
    for train_index, test_index in split:
        # Perform nested training.
        train_screens = dataset[train_index]
        train_pages = [screen.video.pages for screen in train_screens]
        train_observations = list(zip(train_screens, train_pages))
        train_ground_truth = [[page in screen.matching_pages for page in pages] \
                               for screen, pages in train_observations]
        model.fit(train_observations, train_ground_truth)
        # Perform nested testing.
        test_screens = dataset[test_index]
        test_pages = [screen.video.pages for screen in test_screens]
        test_observations = list(zip(test_screens, test_pages))
        test_ground_truth = [[page in screen.matching_pages for page in pages] \
                              for screen, pages in test_observations]
        predictions = model.predict(test_observations)
        loss = zero_one_loss([1 if any(relevance_judgements) else 0 \
                              for relevance_judgements in test_ground_truth],
                             [1 if cls else 0 for cls in predictions])
        losses.append(loss)
    return CrossValidationResult(model, losses)

class Task1Model(object):
    def fit(self, observations, ground_truth):
        """Trains the model using the provided screens and document pages and a ground truth.

        Parameters:
            observations    The provided list of observations in the form of 2-tuples, where each
                            tuple consists of a screen and a list of document pages.
            ground_truth    The provided list of relevance judgements in the form of lists, where
                            each list corresponds to a single observed screen and each list element
                            is a boolean value that corresponds to a single document page.

                            In the case of a subtask A model, these relevance judgements directly
                            correspond to the model's predictions. In the case of a subtask B
                            model, the model is expected to predict True if there is any matching
                            page, e.g. is there is any value True in a given list.
        """
        pass

class Task1SubtaskAModel(Task1Model):
    """ This mixin represents a task 1, subtask A model."""
    def predict(self, observations):
        """Predicts the rankings of the provided document pages based on their similarity to
        provided screens and returns the predicted rankings as an iterable.

        Parameters:
            observations    The provided list of 2-tuples, where each tuple consists of a screen and
                            a list of document pages.
        """
        raise NotImplementedError()

class Task1SubtaskBModel(Task1Model):
    """ This mixin represents a task 1, subtask B model."""
    def predict(self, observations):
        """Predicts whether provided screens display any of the provided document pages and returns
        the predictions as an iterable of truthy and falsy values.

        Parameters:
            observations    The provided list of 2-tuples, where each tuple consists of a screen and
                            a list of document pages.
        """
        raise NotImplementedError()

class CrossValidationResult(object):
    """ This class represents a cross-validation result. """
    def __init__(self, author, results):
        """Constructs a cross-validation result.

        Parameters:
            author  The model that produced the result.
            results A list of results, one for each cross-valudation fold.
        """
        self.author = author
        self.results = results
        self.mean = mean(self.results)

    def confidence_interval(self, confidence=95):
        """
            Computes the empirical confidence interval (sample quantiles) at
            the given confidence level.
        """
        assert confidence >= 0 and confidence <= 100
        return (percentile(self.results, 100-confidence),
                percentile(self.results, confidence))

    def __lt__(self, other):
        return self.mean < other.mean

    def __repr__(self):
        interval = self.confidence_interval()
        return "%s:\tCI: [%0.10f, %0.10f], mean: %10f" % (self.author, interval[0], interval[1],
                                                          self.mean)
