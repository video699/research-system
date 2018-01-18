"""
    This module implements a supervised task 1, subtask A model that trains general pattern
    recognition classifiers on feature vectors extracted from the individual images.
"""

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
from preprocessing import FEATURE_KWARGS, Features

LOGGER = logging.getLogger(__name__)

FEATURE_VECTOR_DTYPE = "float64"
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

class FeatureVectorsClassifier(Model):
    """
        This class represents a task 1, subtask A model that trains pattern recognition classifiers
        on feature vectors extracted from the individual images.
    """

    def __init__(self, classifier, kwargs):
        """Constructs a supervised task1, subtask A model based on pattern recognition classifiers.

        Parameters:
            classifier      The provided supervised classifier.

            kwargs          The parameters for feature generation.
        """
        assert classifier in CLASSIFIERS
        self.classifier = classifier
        self.features = Features(**kwargs)
        self.scaler = StandardScaler()
        self.feature_preselector = SelectKBest(chi2, k=NUM_FEATURES)
        self.feature_selector = RFECV(self.classifier, scoring=SCORING,
                                      cv=StratifiedKFold(NUM_FOLDS, random_state=RANDOM_STATE),
                                      n_jobs=-1)

    def fit(self, videos):
        LOGGER.debug("Preparing training samples for %s ...", self)
        X = []
        y = []
        for video_num, video in enumerate(videos):
            LOGGER.debug("Processing video number %d / %d ...", video_num + 1, len(videos))
            for screen in video.screens:
                for page in video.pages:
                    LOGGER.debug("Processing (%s, %s) ...", page, screen)
                    X.append(self.features.get_pairwise_features(page, screen))
                    y.append(1 if page in screen.matching_pages else 0)
                    LOGGER.debug("Done processing (%s, %s).", page, screen)
            LOGGER.debug("Done processing video number %d / %d.", video_num + 1, len(videos))
        LOGGER.debug("Done preparing training samples for %s.", self)

        LOGGER.debug("Fitting the feature preselector (%d samples, %d features) ...", len(X), len(X[0]))
        self.feature_preselector.fit(X, y)
        X = self.feature_preselector.transform(X)
        LOGGER.debug("Done fitting the feature preselector (%d features).", X.shape[1])

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
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                X = []
                for page in pages:
                    LOGGER.debug("Processing %s ...", page)
                    X.append(self.features.get_pairwise_features(page, screen))
                    LOGGER.debug("Done processing %s.", page)
                X = self.feature_preselector.transform(X)
                X = self.scaler.transform(X)
                if self.classifier.__class__ != SVC:
                    X = self.feature_selector.transform(X)
                ranking = self._predict_confidence(X)
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _predict_confidence(self, X):
        """Produces confidence scores for class 1 for each of the provided feature vectors.

        Parameters:
            X   The list of provided feature vectors."""
        assert "decision_function" in dir(self.classifier) \
            or "predict_proba" in dir(self.classifier)
        if "decision_function" in dir(self.classifier):
            confidence = self.classifier.decision_function(X)
        else:
            confidence = self.classifier.predict_proba(X)[:,1]
        return confidence

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, self.features.__repr__(),
                                self.classifier.__class__.__name__)

    def __repr__(self):
        return "Feature vectors classifier (%s, %s)" % (self.features,
                                                        self.classifier.__class__.__name__)

FEATURE_VECTOR_CLASSIFIERS = [FeatureVectorsClassifier(classifier, kwargs) \
                              for kwargs in FEATURE_KWARGS \
                              for classifier in CLASSIFIERS]
