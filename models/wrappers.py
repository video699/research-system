"""
    This module implements a task 1, subtask B model that uses an arbitrary classifier and the
    scores predicted by an arbitrary task 1, subtask A model to predict whether a screen displays a
    document page from a set or not.
"""

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .base import Task1Model, Task1SubtaskAModel, Task1SubtaskBModel, RANDOM_STATE

WRAPPER_CLASSIFIERS = [SVC(random_state=RANDOM_STATE),
                       LogisticRegression(random_state=RANDOM_STATE),
                       KNeighborsClassifier(n_neighbors=1)]

class Wrapper(Task1SubtaskBModel):
    """
        This class represents a task 1, subtask B model that uses an arbitrary supervised classifier
        and an arbitrary task 1, subtask A model to predict whether a screen displays a document
        page from a set or not.
    """
    def __init__(self, classifier, subtaska_model):
        """Constructs a task 1, subtask B model that uses an arbitrary supervised classifier and an
        arbitrary task 1, subtask A model to predict whether a screen displays a document page from
        a set or not.

        Parameters:
            classifier      The provided supervised classifier.
            subtaska_model  The provided supervised task 1, subtask A model.
        """
        assert isinstance(subtaska_model, Task1SubtaskAModel)
        self.subtaska_model = subtaska_model
        self.is_supervised = self.subtaska_model.fit.__code__ != Task1Model.fit.__code__
        assert isinstance(classifier, ClassifierMixin)
        self.classifier = classifier

    def fit(self, observations, ground_truth):
        if self.is_supervised:
            tune_observations = observations[:len(observations)//2]
            tune_ground_truth = ground_truth[:len(ground_truth)//2]
            self.subtaska_model.fit(tune_observations, tune_ground_truth)
            train_observations = observations[len(observations)//2:]
            train_ground_truth = ground_truth[len(ground_truth)//2:]
        else:
            train_observations = observations
            train_ground_truth = ground_truth
        predictions = self.subtaska_model.predict(train_observations)
        X, Y = [], []
        for ranking, relevance_judgements in zip(predictions, train_ground_truth):
            X.append([sorted(ranking, reverse=True)[0]])
            Y.append(1 if any(relevance_judgements) else 0)
        self.classifier.fit(X, Y)

    def predict(self, observations):
        predictions = self.subtaska_model.predict(observations)
        return self.classifier.predict([[sorted(ranking, reverse=True)[0]] \
                                        for ranking in predictions])

    def _filename(self):
        return "%s.%s.%s-%s__%s" % (super(Wrapper, self)._filename(), __name__,
                                    self.__class__.__name__, self.classifier.__class__.__name__,
                                    self.subtaska_model._filename())

    def __repr__(self):
        wrapper_type = "Supervised" if self.is_supervised else "Unsupervised"
        return "%s wrapper (%s, %s)" % (wrapper_type, self.classifier.__class__.__name__,
                                        self.subtaska_model)
