"""
    This module defines the basic mixins implemented by models and the low-level evaluation
    functions invoked by the individual multiprocessing workers.
"""

from pickle import load, dump
import random

from filenames import RESULT_DIRNAME

from numpy import mean, percentile
from sklearn.metrics import zero_one_loss

RANDOM_STATE = 12345

def evaluate_task1_subtaska_worker(args):
    """Evaluates the provided task 1, subtask A model and returns the average rank of the matching
    document page.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    model, (dataset, split) = args
    result_filename = "%s/%s.pkl" % (RESULT_DIRNAME, model._filename())
    try:
        with open(result_filename, "rb") as f:
            result = load(f)
    except FileNotFoundError:
        random.seed(RANDOM_STATE)
        average_ranks = []
        for train_index, test_index in split:
            train_videos = dataset[train_index]
            model.fit(train_videos)
            test_videos = dataset[test_index]
            test_screens = [screen for video in test_videos for screen in video.screens]
            predictions = model.predict(test_videos)
            ranks = []
            for screen, ranking in zip(test_screens, predictions):
                if not screen.matching_pages:
                    continue
                relevance_judgements = (page in screen.matching_pages \
                                        for page in screen.video.pages)
                _, ranked_relevance_judgements = zip(*sorted(zip(ranking, relevance_judgements),
                                                             reverse=True))
                ranks.append(ranked_relevance_judgements.index(True))
            average_ranks.append(mean(ranks))
        result = CrossValidationResult(model, average_ranks)
        with open(result_filename, "wb") as f:
            dump(result, f)
    return result

def evaluate_task1_subtaskb_worker(args):
    """Evaluates the provided task 1, subtask B model and returns the average rank of the matching
    document page.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    model, (dataset, split) = args
    result_filename = "%s/%s.pkl" % (RESULT_DIRNAME, model._filename())
    try:
        with open(result_filename, "rb") as f:
            result = load(f)
    except FileNotFoundError:
        random.seed(RANDOM_STATE)
        losses = []
        for train_index, test_index in split:
            train_videos = dataset[train_index]
            model.fit(train_videos)
            test_videos = dataset[test_index]
            test_screens = [screen for video in test_videos for screen in video.screens]
            predictions = model.predict(test_videos)
            loss = zero_one_loss([1 if screen.matching_pages else 0 \
                                  for screen in test_screens],
                                 [1 if cls else 0 for cls in predictions])
            losses.append(loss)
        result = CrossValidationResult(model, losses)
        with open(result_filename, "wb") as f:
            dump(result, f)
    return result

class Task1Model(object):
    def fit(self, videos):
        """Trains the model using the provided videos.

        Parameters:
            videos          The provided list of videos.
        """
        pass

    def _filename(self):
        """Returns the filename that serves as a basename to store data related to this model."""
        return "task1"

class Task1SubtaskAModel(Task1Model):
    """ This mixin represents a task 1, subtask A model."""
    def predict(self, videos):
        """For each screen in the provided videos, predicts the ranking of the document pages
           attached to the video based on their similarity to the screen and returns the predicted
           rankings as an iterable.

        Parameters:
            videos          The provided list of videos.
        """
        raise NotImplementedError()

    def _filename(self):
        return "%s-subtaska" % super(Task1SubtaskAModel, self)._filename()

class Task1SubtaskBModel(Task1Model):
    """ This mixin represents a task 1, subtask B model."""
    def predict(self, videos):
        """For each screen in the provided videos, predicts whether any of the document pages
           attached to the video match the screen and returns the prediction as an iterable of
           truthy and falsy values.

        Parameters:
            videos          The provided list of videos.
        """
        raise NotImplementedError()

    def _filename(self):
        return "%s-subtaskb" % super(Task1SubtaskBModel, self)._filename()

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
        return "%s:\tCI: [%0.10f, %0.10f], mean: %0.10f" % (self.author, interval[0], interval[1],
                                                            self.mean)
