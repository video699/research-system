"""
    This module implements a supervised task 1, subtask B model that trains general pattern
    recognition classifiers on scores produced by task 1, subtask A models (regressors).
"""

from joblib import Parallel, delayed
import logging

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, chi2

from .base import Model
from dataset import Page, Screen, RANDOM_STATE
from models.task1.subtaska import VGGS256, ROW_NUMBERS, HISTOGRAMS

LOGGER = logging.getLogger(__name__)

SCORING = "roc_auc"
NUM_FOLDS = 3
NUM_FEATURES = 100

ESTIMATORS = [LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced"),
              SVC(random_state=RANDOM_STATE, class_weight="balanced")]
ENSEMBLES = [AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE,
                                                       class_weight="balanced")),
             RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")]
CLASSIFIERS = ESTIMATORS + ENSEMBLES

PARAM_GRIDS = {
    SVC: [
        {
            "kernel": ["linear"],
            "C": [2**(2*(k-2)-1) for k in range(10)],
        }, {
            "kernel": ["rbf"],
            "C": [2**(2*(k-2)-1) for k in range(10)],
            "gamma": [2**(2*(k-7)-1) for k in range(10)],
        },
    ], AdaBoostClassifier: {
        "n_estimators": [2**(k+2) for k in range(7)],
    }, RandomForestClassifier: {
        "n_estimators": [2**(k+2) for k in range(7)],
        "max_depth": [None] + [2**k for k in range(7)],
    },
}

class RegressorClassifier(Model):
    """
        This class represents a task 1, subtask B model that trains pattern recognition classifiers
        on scores produced by task 1, subtask A models (regressors).
    """

    def __init__(self, classifier, use_vgg256):
        """Constructs a supervised task1, subtask B model based on task 1, subtask A models.

        Parameters:
            classifier      The provided supervised classifier.

            use_vgg256      Whether the VGG256 features should be used.
        """
        assert classifier in CLASSIFIERS
        self.classifier = classifier
        assert type(use_vgg256) is bool
        self.use_vgg256 = use_vgg256
        self.scaler = StandardScaler()
#       self.feature_preselector = SelectKBest(chi2, k=NUM_FEATURES)
        self.feature_selector = RFECV(self.classifier, scoring=SCORING,
                                      cv=StratifiedKFold(NUM_FOLDS, random_state=RANDOM_STATE),
                                      n_jobs=-1)

    def fit(self, videos):
        LOGGER.debug("Preparing training samples for %s ...", self)
        observations = [(screen_video, page_video) \
                        for screen_video in videos for page_video in videos]
        X = self._get_regressor_predictions(observations)
        y = [1 if screen_video == page_video and screen.matching_pages else 0 \
             for screen_video, page_video in observations \
             for screen in screen_video.screens]
        LOGGER.debug("Done preparing training samples for %s.", self)

#       LOGGER.debug("Fitting the feature preselector (%d samples, %d features) ...", len(X), len(X[0]))
#       self.feature_preselector.fit(X, y)
#       X = self.feature_preselector.transform(X)
#       LOGGER.debug("Done fitting the feature preselector (%d features).", X.shape[1])

        LOGGER.debug("Fitting the feature scaler ...")
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        LOGGER.debug("Done fitting the feature scaler.")

        if self.classifier.__class__ != SVC:
            LOGGER.debug("Fitting the feature selector (%d samples, %d features) ...", *X.shape)
            self.feature_selector.fit(X, y)
            X = self.feature_selector.transform(X)
            LOGGER.debug("Done fitting the feature selector. (%d features)", X.shape[1])

        if self.classifier.__class__ in PARAM_GRIDS and self.classifier.__class__ != SVC:
            LOGGER.debug("Optimizing the classifier parameters and fitting the classifier ...")
            param_grid = PARAM_GRIDS[self.classifier.__class__]
            optimizer = GridSearchCV(self.classifier, param_grid, scoring=SCORING, refit=True,
                                     cv=StratifiedKFold(NUM_FOLDS, random_state=RANDOM_STATE))
            optimizer.fit(X, y)
            self.classifier = optimizer.best_estimator_
            LOGGER.debug("Done optimizing the classifier parameters and fitting the classifier.")
        else:
            LOGGER.debug("Fitting the classifier ...")
            self.classifier.fit(X, y)
            LOGGER.debug("Done fitting the classifier.")

    def predict(self, observations):
        X = self._get_regressor_predictions(observations)
#       X = self.feature_preselector.transform(X)
        X = self.scaler.transform(X)
        if self.classifier.__class__ != SVC:
            X = self.feature_selector.transform(X)
        y = self.classifier.predict(X)
        return y

    def _get_regressor_predictions_worker(self, regressor_num, regressor, regressors, observations):
        LOGGER.info("Retrieving rankings from regressor number %d / %d (%s) ...",
                    regressor_num + 1, len(regressors), regressor)
        rankings = regressor.predict(observations)
        scores = [ranking[0] for ranking in rankings]
        LOGGER.info("Done retrieving rankings from regressor number %d / %d.",
                    regressor_num + 1, len(regressors))
        return scores

    def _get_regressor_predictions(self, observations):
        """Predicts the vector of scores assigned to the top-ranking document page in a provided
        video for all screens in another provided video using an ensemble of task 1, subtask A
        models (regressors) and returns the vectors as a list.

        Parameters:
            observations    The provided list of 2-tuples, where each tuple consists of a video
                            containing screens and a video containing document pages.
        """
        LOGGER.debug("Predicting the vector of scores for %s ...", self)
        regressors = VGGS256 + ROW_NUMBERS + HISTOGRAMS if self.use_vgg256 \
            else ROW_NUMBERS + HISTOGRAMS
        predictions = Parallel(n_jobs=-1)(delayed(self._get_regressor_predictions_worker)\
                                                 (regressor_num, regressor, regressors,
                                                  observations) \
                                          for regressor_num, regressor in enumerate(regressors))
#       predictions = []
#       for regressor_num, regressor in enumerate(regressors):
#           LOGGER.info("Retrieving rankings from regressor number %d / %d (%s) ...",
#                       regressor_num + 1, len(regressors), regressor)
#           rankings = regressor.predict(observations)
#           scores = [ranking[0] for ranking in rankings]
#           predictions.append(scores)
#           LOGGER.info("Done retrieving rankings from regressor number %d / %d.",
#                       regressor_num + 1, len(regressors))
        LOGGER.debug("Done predicting the vector of scores for %s.", self)
        return list(zip(*predictions))

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, self.use_vgg256,
                                self.classifier.__class__.__name__)

    def __repr__(self):
        return "Regressor classifier (with%s VGG256, %s)" % ("" if self.use_vgg256 else "out",
                                                             self.classifier.__class__.__name__)

REGRESSOR_CLASSIFIERS = [RegressorClassifier(classifier, use_vgg256) \
                         for use_vgg256 in (True, False) \
                         for classifier in CLASSIFIERS]
