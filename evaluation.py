"""
    This module contains the top-level evaluation routine.
"""
from itertools import chain
import logging
from math import sqrt
from pickle import load, dump
import random

from joblib import Parallel, delayed
from numpy import mean, percentile
from sklearn.model_selection import KFold
from scipy.stats import norm

from dataset import FOLDS_NUM, RANDOM_STATE
from filenames import RESULTS_DIRNAME
from models import TASK1_SUBTASKA_MODELS, TASK1_SUBTASKB_MODELS, Task1Model, Task1SubtaskAModel, \
    Task1SubtaskBModel, TASK1_SUBTASKA_DUMMY, TASK1_SUBTASKB_DUMMY

LOGGER = logging.getLogger(__name__)

def evaluate_task1_subtaska_worker(args):
    """Evaluates the provided task 1, subtask A model and returns the average rank of the matching
    document page.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    model, (dataset, split) = args
    split_hash = hash((tuple(dataset), tuple([(tuple(train_index), tuple(test_index)) \
                                             for train_index, test_index in split])))
    result_filename = "%s/%s-%s.pkl" % (RESULTS_DIRNAME, split_hash, model._filename())
    try:
        with open(result_filename, "rb") as f:
            result = load(f)
    except FileNotFoundError:
        random.seed(RANDOM_STATE)
        ranks = []
        for split_num, (train_index, test_index) in enumerate(split):
            LOGGER.info("Performing model selection at split number %d / %d ...",
                        split_num + 1, len(split))
            train_videos = dataset[train_index]
            LOGGER.info("Fitting %s.", model)
            model.fit(train_videos)
            LOGGER.info("Done fitting %s.", model)

            test_videos = dataset[test_index]
            test_screens = (screen for video in test_videos for screen in video.screens)
            observations = [(video, video) for video in test_videos]
            LOGGER.info("Predicting with %s.", model)
            rankings = model.predict(observations)
            LOGGER.info("Done predicting with %s.", model)
            for screen, ranking in zip(test_screens, rankings):
                if not screen.matching_pages:
                    continue
                relevance_judgements = (page in screen.matching_pages \
                                        for page in screen.video.pages)
                _, ranked_relevance_judgements = zip(*sorted(zip(ranking, relevance_judgements),
                                                             reverse=True))
                ranks.append(ranked_relevance_judgements.index(True))
            LOGGER.info("Done performing model selection at split number %d / %d.",
                        split_num + 1, len(split))
        result = Task1SubtaskACrossValidationResult(model, ranks)
        with open(result_filename, "wb") as f:
            dump(result, f)
    return result

#def evaluate_task1_subtaskb_subworker(model, dataset, train_index, test_index):
#   LOGGER.info("Fitting %s ...", model)
#   train_videos = dataset[train_index]
#   LOGGER.info("Done fitting %s.", model)
#   model.fit(train_videos)
#
#   test_videos = dataset[test_index]
#   observations = [(video, video) for video in test_videos]
#   ground_truth = (1 if screen.matching_pages else 0 \
#                   for video in test_videos for screen in video.screens)
#   LOGGER.info("Predicting with %s ...", model)
#   predictions = (1 if cls else 0 for cls in model.predict(observations))
#   measurements = zip(ground_truth, predictions)
#   LOGGER.info("Done predicting with %s.", model)
#   return list(measurements)

def evaluate_task1_subtaskb_worker(args):
    """Evaluates the provided task 1, subtask B model and returns the misclassification rate.

    Parameters:
        args    A 2-tuple containing the model itself and another 2-tuple containing the task 1
                dataset and its k-fold split.
    """
    model, (dataset, split) = args
    split_hash = hash((tuple(dataset), tuple([(tuple(train_index), tuple(test_index)) \
                                             for train_index, test_index in split])))
    result_filename = "%s/%s-%s.pkl" % (RESULTS_DIRNAME, split_hash, model._filename())
    try:
        with open(result_filename, "rb") as f:
            result = load(f)
    except FileNotFoundError:
        random.seed(RANDOM_STATE)
#       measurements = Parallel(n_jobs=-1)(delayed(evaluate_task1_subtaskb_subworker)\
#                                                 (model, dataset, train_index, test_index) \
#                                          for train_index, test_index in split)
#       result = Task1SubtaskBCrossValidationResult(model, list(chain.from_iterable(measurements)))
        measurements = []
        for split_num, (train_index, test_index) in enumerate(split):
            LOGGER.info("Performing model selection at split number %d / %d ...",
                         split_num + 1, len(split))
            train_videos = dataset[train_index]
            LOGGER.info("Fitting %s ...", model)
            model.fit(train_videos)
            LOGGER.info("Done fitting %s.", model)
 
            test_videos = dataset[test_index]
            observations = [(video, video) for video in test_videos]
            ground_truth = (1 if screen.matching_pages else 0 \
                            for video in test_videos for screen in video.screens)
            LOGGER.info("Predicting with %s ...", model)
            predictions = (1 if cls else 0 for cls in model.predict(observations))
            measurements.extend(zip(ground_truth, predictions))
            LOGGER.info("Done predicting with %s.", model)
            LOGGER.info("Done performing model selection at split number %d / %d.",
                         split_num + 1, len(split))
        result = Task1SubtaskBCrossValidationResult(model, measurements)
        with open(result_filename, "wb") as f:
            dump(result, f)
    return result

class CrossValidationResult(object):
    """ This class represents a cross-validation result. """
    def __init__(self, model, predictions):
        """Constructs a cross-validation result.

        Parameters:
            model         The model that produced the predictions.
            predictions   An iterable of predictions produced by a model.
        """
        assert isinstance(model, Task1Model)
        self.model_filename = model._filename()
        self.model_repr = model.__repr__()
        self.predictions = list(predictions)

    def __repr__(self):
        return "%s:\t" % self.model_repr

class Task1SubtaskACrossValidationResult(CrossValidationResult):
    """ This class represents a task 1, subtask A cross-validation result. """
    def __init__(self, model, ranks):
        assert isinstance(model, Task1SubtaskAModel)
        self.mean = mean(ranks)
        super(Task1SubtaskACrossValidationResult, self).__init__(model, ranks)

    def confidence_interval(self, alpha=0.05):
        """
            Computes the empirical confidence interval (sample quantiles) at
            the given confidence level.
        """
        assert alpha >= 0 and alpha <= 1
        return (percentile(self.predictions, 100*alpha/2),
                percentile(self.predictions, 100*(1-alpha/2)))

    def __lt__(self, other):
        return self.mean < other.mean

    def __repr__(self):
        interval = self.confidence_interval()
        return "%smean rank: %0.10f, CI: [%0.10f, %0.10f]" \
            % (super(Task1SubtaskACrossValidationResult, self).__repr__(),
               self.mean, interval[0], interval[1])

class Task1SubtaskBCrossValidationResult(CrossValidationResult):
    """ This class represents a task 1, subtask B cross-validation result. """
    def __init__(self, model, predictions):
        assert isinstance(model, Task1SubtaskBModel)
        self.error_rate = \
            sum((cls_true != cls_pred for cls_true, cls_pred in predictions)) / len(predictions)
        super(Task1SubtaskBCrossValidationResult, self).__init__(model, predictions)

    def confidence_interval(self, alpha=0.05):
        """
            Estimates the confidence interval by taking the normal distribution quantiles at the
            given confidence level.
        """
        assert alpha >= 0 and alpha <= 1
        z = norm.ppf(1 - alpha/2)
        p = self.error_rate
        n = len(self.predictions)
        size = z * sqrt((p*(1-p))/n)
        return (max(0, p-size), min(1, p+size))

    def __lt__(self, other):
        return self.error_rate < other.error_rate

    def __repr__(self):
        interval = self.confidence_interval()
        return "%serror rate: %0.10f, CI: [%0.10f, %0.10f]" \
            % (super(Task1SubtaskBCrossValidationResult, self).__repr__(),
               self.error_rate, interval[0], interval[1])

def evaluate_task1_subtaska(dataset):
    """Performs the task 1, subtask A evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    LOGGER.info("Evaluating subtask A (screen-based document page retrieval) ...")

#   LOGGER.info("Performing performance estimation ...")
    outer_kfold = KFold(n_splits=FOLDS_NUM, shuffle=True, random_state=RANDOM_STATE)
    outer_split = list(outer_kfold.split(dataset))
#   for train_index, test_index in outer_split:
    train_index, test_index = outer_split[0]

    LOGGER.info("Performing model selection ...")
    inner_kfold = KFold(n_splits=FOLDS_NUM-1, shuffle=True, random_state=RANDOM_STATE)
    inner_split = list(inner_kfold.split(dataset[train_index]))
    results = []
    for model in TASK1_SUBTASKA_MODELS:
        args = (model, (dataset[train_index], inner_split))
        result = evaluate_task1_subtaska_worker(args)
        results.append((result, model))
    sorted_results = sorted(results, reverse=False, key=lambda x: x[0])
    LOGGER.info("Model selection results:")
    for result, _ in sorted_results:
        LOGGER.info("- %s", result)
    LOGGER.info("Done performing model selection.")

#   TODO: Print the performance estimates for the best models selected in the individual rounds.
    LOGGER.info("Performing performance estimation ...")
    best_model = [model for _, model in sorted_results if model not in TASK1_SUBTASKA_DUMMY]
    results = []
    for model in [best_model[0]] + TASK1_SUBTASKA_DUMMY:
        args = (model, (dataset, [(train_index, test_index)]))
        result = evaluate_task1_subtaska_worker(args)
        results.append((result, model))
    sorted_results = sorted(results, reverse=False, key=lambda x: x[0])
    LOGGER.info("Performance estimation result:")
    for result, _ in sorted_results:
        LOGGER.info("- %s", result)
    LOGGER.info("Done performing performance estimation.")

    LOGGER.info("Done evaluating subtask A.")

def evaluate_task1_subtaskb(dataset):
    """Performs the task 1, subtask B evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    LOGGER.info("Evaluating subtask B (screen-based document match detection) ...")

#   LOGGER.info("Performing performance estimation ...")
    outer_kfold = KFold(n_splits=FOLDS_NUM, shuffle=True, random_state=RANDOM_STATE)
    outer_split = list(outer_kfold.split(dataset))
#   for train_index, test_index in outer_split:
    train_index, test_index = outer_split[0]

    LOGGER.info("Performing model selection ...")
    inner_kfold = KFold(n_splits=FOLDS_NUM-1, shuffle=True, random_state=RANDOM_STATE)
    inner_split = list(inner_kfold.split(dataset[train_index]))
    results = []
    for model in TASK1_SUBTASKB_MODELS:
        args = (model, (dataset[train_index], inner_split))
        result = evaluate_task1_subtaskb_worker(args)
        results.append((result, model))
    sorted_results = sorted(results, reverse=False, key=lambda x: x[0])
    LOGGER.info("Model selection results:")
    for result, _ in sorted_results:
        LOGGER.info("- %s", result)
    LOGGER.info("Done performing model selection.")

#   TODO: Print the performance estimates for the best models selected in the individual rounds.
    LOGGER.info("Performing performance estimation ...")
    best_model = [model for _, model in sorted_results if model not in TASK1_SUBTASKB_DUMMY]
    results = []
    for model in [best_model[0]] + TASK1_SUBTASKB_DUMMY:
        args = (model, (dataset, [(train_index, test_index)]))
        result = evaluate_task1_subtaskb_worker(args)
        results.append((result, model))
    sorted_results = sorted(results, reverse=False, key=lambda x: x[0])
    LOGGER.info("Performance estimation result:")
    for result, _ in sorted_results:
        LOGGER.info("- %s", result)
    LOGGER.info("Done performing performance estimation.")

    LOGGER.info("Done evaluating subtask B.")

def evaluate_task1(dataset):
    """Performs the task 1 evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    LOGGER.info("Evaluating task 1 ...")
    evaluate_task1_subtaska(dataset)
    evaluate_task1_subtaskb(dataset)
    LOGGER.info("Done evaluating task 1.")
