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

    def fit(self, videos):
        if self.is_supervised:
            tune_videos = videos[:len(videos)//2]
            self.subtaska_model.fit(tune_videos)
            train_videos = videos[len(observations)//2:]
        else:
            train_videos = videos
        predictions = self.subtaska_model.predict(train_videos)
        X, Y = [], []
        for video in train_videos:
            pages = video.pages
            for ranking, screen in zip(predictions, video.screens):
                X.append([sorted(ranking, reverse=True)[0]])
                Y.append(1 if screen.matching_pages else 0)
        self.classifier.fit(X, Y)

    def predict(self, videos):
        predictions = self.subtaska_model.predict(videos)
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
