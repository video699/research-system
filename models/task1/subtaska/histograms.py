"""
    This module implements task 1, subtask A models that ranks pairs of images based on the distance
    / similarity of their various histograms.
"""

import logging

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .base import Model
from preprocessing import FEATURE_KWARGS, Features, IMAGE_KWARGS, Image

HISTOGRAM_DISTANCES = {"Bhattacharyya distance": cv2.HISTCMP_BHATTACHARYYA,
                       "Kullback–Leibler divergence": cv2.HISTCMP_KL_DIV,
                       "Chi^2 test": cv2.HISTCMP_CHISQR}
HISTOGRAM_SIMILARITIES = {"Correlation": cv2.HISTCMP_CORREL,
                          "Intersection": cv2.HISTCMP_INTERSECT,
                          "Cosine similarity": cosine_similarity}
HISTOGRAM_MEASURES = dict(HISTOGRAM_DISTANCES, **HISTOGRAM_SIMILARITIES)

LOGGER = logging.getLogger(__name__)

class Histograms(Model):
    """
        This class represents a task 1, subtask A model that compares histograms.
    """

    def __init__(self, measure):
        """Constructs an un supervised task1, subtask A model that compares histograms.

        Parameters:
            measure     The OpenCV histogram distance / similarity measure that will be used to
                        compare histograms.
        """
        assert measure in HISTOGRAM_MEASURES
        self.measure = measure

    def _compare_histograms(self, first, second):
        """Measutes the distance / similarity between two provided histograms.

        Parameters:
            first   The first provided histogram.
            second  The second provided histogram."""
        if isinstance(HISTOGRAM_MEASURES[self.measure], int):
            score = cv2.compareHist(first.astype("float32"), second.astype("float32"),
                                    HISTOGRAM_MEASURES[self.measure])
        else:
            score = HISTOGRAM_MEASURES[self.measure]([first], [second])
        assert np.isfinite(score)
        return score

class RowHeights(Histograms):
    """
        This class represents a task 1, subtask A model that ranks pairs of images based on the
        difference in their row heights.
    """

    def __init__(self, image_kwargs, *args, **kwargs):
        """Constructs an un supervised task1, subtask A model that compares row height histograms.

        Parameters:
            image_kwargs    The parameters for the preprocessing of images.

            *args           The arguments that will be passed to the  superclass constructor.

            **kwargs        The keyword arguments that will be passed to the  superclass constructor.
        """
        self.kwargs = image_kwargs
        super(RowHeights, self).__init__(*args, **kwargs)

    def predict(self, observations):
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                screen_histogram = Image(screen, **self.kwargs).get_rows()[3][0][0]
                ranking = []
                for page in pages:
                    LOGGER.debug("Processing %s ...", page)
                    page_histogram = Image(page, **self.kwargs).get_rows()[3][0][0]
                    score = self._compare_histograms(screen_histogram, page_histogram)
                    ranking.append(score if self.measure in HISTOGRAM_SIMILARITIES else -score)
                    LOGGER.debug("Done processing %s.", page)
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, sorted(self.kwargs.items()),
                                self.measure)

    def __repr__(self):
        return "Row heights (%s, %s)" % (self.kwargs, self.measure)

class RowAltitudes(Histograms):
    """
        This class represents a task 1, subtask A model that ranks pairs of images based on the
        difference in their row altitudes.
    """

    def __init__(self, image_kwargs, *args, **kwargs):
        """Constructs an un supervised task1, subtask A model that compares row altitude histograms.

        Parameters:
            image_kwargs    The parameters for the preprocessing of images.

            *args           The arguments that will be passed to the  superclass constructor.

            **kwargs        The keyword arguments that will be passed to the  superclass constructor.
        """
        self.kwargs = image_kwargs
        super(RowAltitudes, self).__init__(*args, **kwargs)

    def predict(self, observations):
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                screen_histogram = Image(screen, **self.kwargs).get_rows()[3][1][0]
                ranking = []
                for page in pages:
                    LOGGER.debug("Processing %s ...", page)
                    page_histogram = Image(page, **self.kwargs).get_rows()[3][1][0]
                    score = self._compare_histograms(screen_histogram, page_histogram)
                    ranking.append(score if self.measure in HISTOGRAM_SIMILARITIES else -score)
                    LOGGER.debug("Done processing %s.", page)
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__,  self.__class__.__name__, sorted(self.kwargs.items()),
                                self.measure)

    def __repr__(self):
        return "Row altitudes (%s, %s)" % (self.kwargs, self.measure)

class GrayscaleHistograms(Histograms):
    """
        This class represents a task 1, subtask A model that ranks pairs of images based on the
        difference in their grayscale histograms.
    """

    def __init__(self, image_kwargs, *args, **kwargs):
        """Constructs an un supervised task1, subtask A model that compares grayscale histograms.

        Parameters:
            image_kwargs    The parameters for the preprocessing of images.
            *args           The arguments that will be passed to the  superclass constructor.

            **kwargs        The keyword arguments that will be passed to the  superclass constructor.
        """
        self.kwargs = image_kwargs
        super(GrayscaleHistograms, self).__init__(*args, **kwargs)

    def predict(self, observations):
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                screen_histogram = Image(screen, **self.kwargs).get_histogram()[0]
                ranking = []
                for page in pages:
                    LOGGER.debug("Processing %s ...", page)
                    page_histogram = Image(page, **self.kwargs).get_histogram()[0]
                    score = self._compare_histograms(screen_histogram, page_histogram)
                    ranking.append(score if self.measure in HISTOGRAM_SIMILARITIES else -score)
                    LOGGER.debug("Done processing %s.", page)
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, sorted(self.kwargs.items()), \
                                self.measure)

    def __repr__(self):
        return "Grayscale histograms (%s, %s)" % (self.kwargs, self.measure)

class FeatureVectors(Histograms):
    """
        This class represents a task 1, subtask A model that ranks pairs of images based on the
        difference in their feature vectors.
    """

    def __init__(self, feature_kwargs, *args, **kwargs):
        """Constructs a semi-supervised task1, subtask A model that compares feature vectors.

        Parameters:
            feature_kwargs  The parameters for feature generation.

            *args           The arguments that will be passed to the  superclass constructor.

            **kwargs        The keyword arguments that will be passed to the  superclass constructor.
        """
        self.scaler = StandardScaler()
        self.features = Features(**feature_kwargs)
        super(FeatureVectors, self).__init__(*args, **kwargs)

    def fit(self, videos):
        LOGGER.debug("Preparing training samples for %s ...", self)
        X = []
        for video in videos:
            for screen in video.screens:
                LOGGER.debug("Processing %s ...", screen)
                features = self.features.get_full_features(screen)
                X.append(features)
                LOGGER.debug("Done processing %s.", screen)
        LOGGER.debug("Done preparing training samples for %s.", self)

        LOGGER.debug("Fitting the feature scaler ...")
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        LOGGER.debug("Done fitting the feature scaler.")

    def predict(self, observations):
        rankings = []
        for observation_num, (screen_video, page_video) in enumerate(observations):
            LOGGER.debug("Processing observation number %d / %d ...", observation_num + 1,
                         len(observations))
            screens = screen_video.screens
            pages = page_video.pages
            for screen in screens:
                LOGGER.debug("Processing %s ...", screen)
                screen_features = self.features.get_full_features(screen)
                ranking = []
                for page in pages:
                    LOGGER.debug("Processing %s ...", page)
                    page_features = self.features.get_full_features(page)
                    score = self._compare_histograms(page_features, screen_features)
                    ranking.append(score if self.measure in HISTOGRAM_SIMILARITIES else -score)
                    LOGGER.debug("Done processing %s.", page)
                rankings.append(ranking)
                LOGGER.debug("Done processing %s.", screen)
            LOGGER.debug("Done processing observation number %d / %d.", observation_num + 1,
                         len(observations))
        return rankings

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, self.features, \
                                self.measure)

    def __repr__(self):
        return "Feature vectors (%s, %s)" % (self.features, self.measure)

ROW_HEIGHTS = [RowHeights(kwargs, measure) \
               for measure in HISTOGRAM_MEASURES \
               for kwargs in IMAGE_KWARGS]
ROW_ALTITUDES = [RowAltitudes(kwargs, measure) \
                 for measure in HISTOGRAM_MEASURES \
                 for kwargs in IMAGE_KWARGS]
GRAYSCALE_HISTOGRAMS = [GrayscaleHistograms(kwargs, measure) \
                        for measure in HISTOGRAM_MEASURES \
                        for kwargs in IMAGE_KWARGS]
FEATURE_VECTORS = [FeatureVectors(kwargs, measure) \
                   for measure in HISTOGRAM_MEASURES \
                   for kwargs in FEATURE_KWARGS]
