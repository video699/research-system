"""
    This module contains the feature generation routines.
"""

import logging
import lzma
from lzma import FORMAT_XZ
import numpy as np
from pickle import load, dump

from dataset import Page, Screen, Frame
from .images import KWARGS as IMAGE_KWARGS, Image
from filenames import FEATURE_CACHES_FILENAME

LOGGER = logging.getLogger(__name__)

FEATURES_DTYPE = "float64"

CACHES = {
    "full_features": {},
}

def load_caches():
    """
        Loads the feature caches off a persistent storage.
    """
    global CACHES
    try:
        LOGGER.debug("Loading the feature caches ...")
        with lzma.open("%s.pkl.xz" % FEATURE_CACHES_FILENAME, "rb") as f:
            CACHES = load(f)
        LOGGER.debug("Done loading the feature caches.")
    except FileNotFoundError:
        LOGGER.debug("Feature caches not found.")

def dump_caches():
    """
        Dumps the preprocessing caches on a persistent storage.
    """
    LOGGER.debug("Dumping the feature caches ...")
    with lzma.open("%s.pkl.xz" % FEATURE_CACHES_FILENAME, "wb", format=FORMAT_XZ, preset=9) as f:
        dump(CACHES, f)
    LOGGER.debug("Done dumping the feature caches.")

class Features(object):
    """
        This class assigns feature vectors to single images and pairs of images.
    """
    def __init__(self, use_vgg256):
        """Constructs an object that assigns feature vector to.

        Parameters:
            use_vgg256      Whether the VGG256 features should be used.
        """
        assert type(use_vgg256) is bool
        self.use_vgg256 = use_vgg256

    def get_features(self, image):
        """Produces a feature vector for the provided Image object.

        Parameters:
            image      The provided screen Image object."""
        assert isinstance(image, Image)
        rows = image.get_rows()
        row_num = np.array([rows[0]])
        row_heights = rows[3][0][0]
        row_altitudes = rows[3][1][0]
        histogram = image.get_histogram()[0]
        haralick = image.get_haralick().ravel()
        vgg256 = image.obj.vgg256[0] + image.obj.vgg256[1]
        arrays = [row_num, row_heights, row_altitudes, histogram, haralick]
        if self.use_vgg256:
            arrays.append(vgg256)
        return np.concatenate(arrays).astype(FEATURES_DTYPE)

    def get_full_features(self, obj):
        """Produces a full feature vector for the provided Page, Screen, or Frame objects. The
        feature vector will contain features for all permissible image preprocessing parameters.

        Parameters:
            obj     The provided Page, Screen, or Frame object."""
        assert isinstance(obj, Page) or isinstance(obj, Screen) or isinstance(obj, Frame)
        if (self, obj) not in CACHES["full_features"]:
            arrays = []
            for kwargs in IMAGE_KWARGS:
                features = self.get_features(Image(obj, **kwargs))
                arrays.append(features)
            full_features = np.concatenate(arrays).astype(FEATURES_DTYPE)
            CACHES["full_features"][(self, obj)] = full_features
        return CACHES["full_features"][(self, obj)]

    def get_pairwise_features(self, page, screen):
        """Produces a pairwise feature vector for the provided Page and Screen objects.

        Parameters:
            page    The provided Page object.
            screen  The provided Screen object."""
        assert isinstance(page, Page)
        assert isinstance(screen, Screen)
        page_features = self.get_full_features(page)
        screen_features = self.get_full_features(screen)
#       return np.concatenate((page_features, screen_features)).astype(FEATURES_DTYPE)
        return np.abs(page_features - screen_features)

    def __hash__(self):
        return self.use_vgg256.__hash__()

    def __eq__(self, other):
        return isinstance(other, Features) and self.use_vgg256 == other.use_vgg256

    def __repr__(self):
        return "Features with%s VGG256" % ("" if self.use_vgg256 else "out")

# Tunable parameters for the feature generation.
KWARGS = [{"use_vgg256": use_vgg256} for use_vgg256 in (True, False)]
