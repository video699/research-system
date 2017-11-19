"""
    This module contains the top-level evaluation routine.
"""
import logging
from multiprocessing import Pool

from sklearn.model_selection import KFold

from dataset import FOLDS_NUM
from models import evaluate_task1_subtaska_worker, evaluate_task1_subtaskb_worker, \
    TASK1_SUBTASKA_MODELS, TASK1_SUBTASKB_MODELS

LOGGER = logging.getLogger(__name__)
POOL = Pool()

def evaluate_task1_subtaska(dataset):
    """Performs the task 1, subtask A evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    kfold = KFold(n_splits=FOLDS_NUM, shuffle=False)

    LOGGER.info("Evaluating subtask A (screen-based document page retrieval) ...")
    split = list(kfold.split(dataset))
    results = POOL.map(evaluate_task1_subtaska_worker,
                       [(model, (dataset, split)) for model in TASK1_SUBTASKA_MODELS])
#   results = []
#   for model in TASK1_SUBTASKA_MODELS:
#       args = (model, (dataset, split))
#       results.append(evaluate_task1_subtaska_worker(args))
    LOGGER.info("Done evaluating subtask A with the following results:")
    for result in sorted(results, reverse=False):
        LOGGER.info("- %s", result)

def evaluate_task1_subtaskb(dataset):
    """Performs the task 1, subtask B evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    kfold = KFold(n_splits=FOLDS_NUM, shuffle=False)

    LOGGER.info("Evaluating subtask B (screen-based document match detection) evaluation ...")
    split = list(kfold.split(dataset))
    results = POOL.map(evaluate_task1_subtaskb_worker,
                       [(model, (dataset, split)) for model in TASK1_SUBTASKB_MODELS])
#   results = []
#   for model in TASK1_SUBTASKB_MODELS:
#       args = (model, (dataset, split))
#       results.append(evaluate_task1_subtaskb_worker(args))
    LOGGER.info("Done evaluating subtask B with the following results:")
    for result in sorted(results, reverse=False):
        LOGGER.info("- %s", result)

def evaluate_task1(dataset):
    """Performs the task 1 evaluation.

    Parameters:
        dataset The provided task 1 dataset."""
    LOGGER.info("Evaluating task 1 ...")
    evaluate_task1_subtaska(dataset)
    evaluate_task1_subtaskb(dataset)
    LOGGER.info("Done evaluating task 1.")
